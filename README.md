# Approximate Flops

A script to calculate flops approximately (mainly coming from matmul).

You can code the forward computation with FakeTensor and computed flops will be printed, like this

```
register_variable("B", 1)
register_variable("T", 4)
register_variable("H", 2)
register_variable("D", 4)
register_variable("C", "D*H")
register_variable("D*H", "C")

x = FakeTensor("B,T,C")
Wq = FakeTensor("C,C")
Wk = FakeTensor("C,C")
Wv = FakeTensor("C,C")
q = x.matmul(Wq).view("B,T,H,D").transpose(1,2)
print(q)
k = x.matmul(Wk).view("B,T,H,D").transpose(1,2)
v = x.matmul(Wv).view("B,T,H,D").transpose(1,2)
# if `causal' is True, the calculated flops will be half-ed
attn = q.matmul(k.transpose(-1,-2), causal=True)
Wo = FakeTensor("C,C")
merged_v = attn.matmul(v, causal=True).view("B,T,C")
x2 = merged_v.matmul(Wo)
fc1 = FakeTensor("C,4*C")
fc2 = FakeTensor("4*C,C")
x3 = x2.matmul(fc1).matmul(fc2)

FakeTensor.print_flops()
```

Running result:
```
Matmul: (B,T,C)-(1,4,8) @ (C,C)-(8,8) -> (B,T,C)-(1,4,8), new_flops=2.0BCCT total_flops=2.0BCCT=512
(B,H,T,D)-(1,2,4,4)
Matmul: (B,T,C)-(1,4,8) @ (C,C)-(8,8) -> (B,T,C)-(1,4,8), new_flops=2.0BCCT total_flops=4.0BCCT=1024
Matmul: (B,T,C)-(1,4,8) @ (C,C)-(8,8) -> (B,T,C)-(1,4,8), new_flops=2.0BCCT total_flops=6.0BCCT=1536
Matmul: (B,H,T,D)-(1,2,4,4) @ (B,H,D,T)-(1,2,4,4) -> (B,H,T,T)-(1,2,4,4), new_flops=BCTT total_flops=6.0BCCT+BCTT=1664
Matmul: (B,H,T,T)-(1,2,4,4) @ (B,H,T,D)-(1,2,4,4) -> (B,H,T,D)-(1,2,4,4), new_flops=BCTT total_flops=6.0BCCT+2.0BCTT=1792
Matmul: (B,T,C)-(1,4,8) @ (C,C)-(8,8) -> (B,T,C)-(1,4,8), new_flops=2.0BCCT total_flops=8.0BCCT+2.0BCTT=2304
Matmul: (B,T,C)-(1,4,8) @ (C,4*C)-(8,32) -> (B,T,4*C)-(1,4,32), new_flops=8.0BCCT total_flops=16.0BCCT+2.0BCTT=4352
Matmul: (B,T,4*C)-(1,4,32) @ (4*C,C)-(32,8) -> (B,T,C)-(1,4,8), new_flops=8.0BCCT total_flops=24.0BCCT+2.0BCTT=6400
TotalFlops: 24.0BCCT+2.0BCTT=6400
```
