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
