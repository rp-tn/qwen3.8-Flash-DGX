# Why vLLM's per-expert weight load is slow on GB10 (docs/HOW-IT-WORKS.md, patch 14).
# Copies every routed-expert weight + block scale of ONE shard to the GPU; run one variant per fresh
# process, each on a shard no earlier run touched (V=variant, F=shard path; needs a free GPU):
#   A  vLLM today: mmap view -> param.copy_() per tensor
#   E  prefetch the shard into the page cache (parallel sequential reads), then A
#   F  .clone() the mmap view into ordinary memory, then the same per-tensor copy_
#   B  reused pinned bounce buffer, one H2D per tensor
#   C  mmap -> pinned staging buffer per 3072 tensors -> one H2D + GPU scatter
import os, sys, time, threading
import torch
from safetensors import safe_open

F, V = os.environ["F"], os.environ["V"]
fd = os.open(F, os.O_RDONLY)
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)          # drop this client's cache for the file
f = safe_open(F, framework="pt")
names = [k for k in f.keys() if ".experts." in k and "shared" not in k
         and (k.endswith("_proj.weight") or k.endswith("_proj.weight_scale"))]
nb = lambda t: t.numel() * t.element_size()
dst = {n: torch.empty(f.get_slice(n).get_shape(), dtype=f.get_tensor(n).dtype, device="cuda") for n in names}
total = sum(nb(d) for d in dst.values())
fsize = os.fstat(fd).st_size
torch.cuda.synchronize()
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)


def prefetch(threads=8, block=16 << 20):
    step = (fsize + threads - 1) // threads
    def work(i):
        buf = bytearray(block); o, end = i * step, min(fsize, (i + 1) * step)
        while o < end:
            n = os.preadv(fd, [memoryview(buf)[: min(block, end - o)]], o)
            if n <= 0: break
            o += n
    ts = [threading.Thread(target=work, args=(i,)) for i in range(threads)]
    [t.start() for t in ts]; [t.join() for t in ts]


def A():
    for n in names:
        dst[n].copy_(f.get_tensor(n))
    torch.cuda.synchronize()


def B():   # reused pinned bounce buffer, one H2D per tensor (the minimal patch)
    pin = torch.empty(max(nb(d) for d in dst.values()), dtype=torch.uint8).pin_memory()
    s = torch.cuda.current_stream()
    for n in names:
        t = f.get_tensor(n); b = nb(t)
        pin[:b].copy_(t.view(-1).view(torch.uint8))
        dst[n].view(-1).view(torch.uint8).copy_(pin[:b], non_blocking=True)
        s.synchronize()


def Fv():  # plain CPU clone first (ordinary malloc'd memory), then the same per-tensor copy_
    for n in names:
        dst[n].copy_(f.get_tensor(n).clone())
    torch.cuda.synchronize()


def C(batch=1536 * 2):
    big = torch.empty(max(sum(nb(dst[n]) for n in names[i:i + batch]) for i in range(0, len(names), batch)),
                      dtype=torch.uint8).pin_memory()
    gbig = torch.empty_like(big, device="cuda")
    for i in range(0, len(names), batch):
        o, spans = 0, []
        for n in names[i:i + batch]:
            t = f.get_tensor(n); b = nb(t)
            big[o:o + b].copy_(t.view(-1).view(torch.uint8)); spans.append((n, o, b)); o += b
        gbig[:o].copy_(big[:o], non_blocking=True)
        for n, o2, b in spans:
            dst[n].view(-1).view(torch.uint8).copy_(gbig[o2:o2 + b])
        torch.cuda.synchronize()


t0 = time.perf_counter()
if V == "A":
    A(); tp = 0.0
elif V == "E":
    prefetch(); tp = time.perf_counter() - t0; A()
elif V == "C":
    C(); tp = 0.0
elif V == "B":
    B(); tp = 0.0
elif V == "F":
    Fv(); tp = 0.0
dt = time.perf_counter() - t0
print(f"{V}  {os.path.basename(F)} ({fsize/2**30:.1f} GiB file, {len(names)} expert tensors, {total/2**30:.1f} GiB): "
      f"{dt:6.1f} s total" + (f" (prefetch {tp:.1f} s)" if tp else "") +
      f"  {dt/len(names)*1e3:.3f} ms/tensor  -> all ~149k expert weight + block-scale tensors ~{dt/len(names)*149e3:.0f} s")
ok = all(torch.equal(dst[n].cpu(), f.get_tensor(n)) for n in names[::997])
print(f"   bytes identical: {ok}")
