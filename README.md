# Approximate Flops/Communications

A script to calculate flops/comms approximately.

You can code the forward computation with FakeTensor and computed flops/comms will be printed, like this

```
register_variable("B", 2)
register_variable("T", 8)
register_variable("H", 2)
register_variable("D", 4)
register_variable("C", "D*H")
register_variable("D*H", "C")

x = FakeTensor("B,T,C").layer_norm()
Wq = FakeTensor("C,C")
Wk = FakeTensor("C,C")
Wv = FakeTensor("C,C")
q = x.matmul(Wq).view("B,T,H,D").transpose(1,2)
print(q)
k = x.matmul(Wk).view("B,T,H,D").transpose(1,2)
v = x.matmul(Wv).view("B,T,H,D").transpose(1,2)
# if `causal' is True, the calculated flops will be half-ed
attn = q.matmul(k.transpose(-1,-2), causal=True).softmax()
Wo = FakeTensor("C,C")
merged_v = attn.matmul(v, causal=True).transpose(1,2).view("B,T,C")
x2 = merged_v.matmul(Wo).layer_norm()
fc1 = FakeTensor("C,4*C")
fc2 = FakeTensor("4*C,C")
x3 = x2.matmul(fc1).matmul(fc2)
```

Running result:
```
LayerNorm: ignored_flops=BCT=128
Matmul: (B,T,C)-(2,8,8) @ (C,C)-(8,8) -> (B,T,C)-(2,8,8) new_flops=2.0BCCT total_flops=2.0BCCT=2048
Matmul: (B,T,C)-(2,8,8) @ (C,C)-(8,8) -> (B,T,C)-(2,8,8) new_flops=2.0BCCT total_flops=4.0BCCT=4096
Matmul: (B,T,C)-(2,8,8) @ (C,C)-(8,8) -> (B,T,C)-(2,8,8) new_flops=2.0BCCT total_flops=6.0BCCT=6144
Matmul: (B,H,T,D)-(2,2,8,4) @ (B,H,D,T)-(2,2,4,8) -> (B,H,T,T)-(2,2,8,8) new_flops=BCTT total_flops=6.0BCCT+BCTT=7168
Softmax: ignored_flops=BHTT=256
Matmul: (B,H,T,T)-(2,2,8,8) @ (B,H,T,D)-(2,2,8,4) -> (B,H,T,D)-(2,2,8,4) new_flops=BCTT total_flops=6.0BCCT+2.0BCTT=8192
Matmul: (B,T,C)-(2,8,8) @ (C,C)-(8,8) -> (B,T,C)-(2,8,8) new_flops=2.0BCCT total_flops=8.0BCCT+2.0BCTT=10240
LayerNorm: ignored_flops=BCT=128
Matmul: (B,T,C)-(2,8,8) @ (C,4*C)-(8,32) -> (B,T,4*C)-(2,8,32) new_flops=8.0BCCT total_flops=16.0BCCT+2.0BCTT=18432
Matmul: (B,T,4*C)-(2,8,32) @ (4*C,C)-(32,8) -> (B,T,C)-(2,8,8) new_flops=8.0BCCT total_flops=24.0BCCT+2.0BCTT=26624
Showing DAG...(may not popup)
TotalFlops: 24.0BCCT+2.0BCTT=26624
IgnoredFlops: 2.0BCT+BHTT=512
TotalComms: <None>=0
```

Call `.show_compute_graph()` on FakeTensors to show the DAG
![example.png](./example.png)

