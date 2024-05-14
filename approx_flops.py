from dataclasses import dataclass

@dataclass
class VarMeta:
    value: float
    reductions: list[str]

g_var_registry = dict[str, VarMeta]()
g_merge_registry = dict[str, VarMeta]()

def parse_shape_names(shapedesc: str):
    return [x.strip() for x in shapedesc.split(',')]

def is_num(s):
    try: float(s)
    except ValueError: return False
    else: return True

def get_atomic_names(var_name: str, float_or_int:bool) -> tuple[float|int, list]:
    if not '*' in var_name:
        # a single variable
        return (1, [var_name])
    else:
        const_factor = 1
        # multiple variable multiplied, e.g. "H*D"
        atomic_names = []
        for sub_var_name in var_name.split('*'):
            sub_var_name = sub_var_name.strip()
            if is_num(sub_var_name): # constant, e.g. the "4" in "4*C"
                const_factor *= float(sub_var_name) if float_or_int else int(sub_var_name)
            else:
                atomic_names.append(sub_var_name)
        return (const_factor, atomic_names)
    
def get_varmeta_from_desc(desc: str):
    cf, remaining_names = get_atomic_names(desc, True)
    p = cf
    for name in remaining_names:
        if not name in g_var_registry:
            raise Exception(f"Unknown desc={name}, register it with register_variable")
        p *= g_var_registry[name].value
    reductions = [str(cf)]
    reductions.extend(remaining_names)
    return VarMeta(p, reductions)

def get_int_from_desc(desc: str) -> int:
    cf, remaining_names = get_atomic_names(desc, False)
    p = cf
    for name in remaining_names:
        if not name in g_var_registry:
            raise Exception(f"Unknown var_name={name}, register it with register_variable")
        p *= g_var_registry[name].value
    assert int(p) == p, f"Expect {p} to be integer"
    return int(p)

# NOTE: could be overwritten
def register_variable(var_name:str, val):
    if isinstance(val, int) or isinstance(val, float):
        g_var_registry[var_name] = VarMeta(val, [])
    elif isinstance(val, str):
        # handy to define variable based on product of other variables
        if not '*' in var_name:
            g_var_registry[var_name] = get_varmeta_from_desc(val)
        else:
            g_merge_registry[var_name] = get_varmeta_from_desc(val)
    else:
        raise Exception("Unsupported type of val")

def get_shape_from_names(shape_names: list[str]):
    return [get_int_from_desc(var_name) for var_name in shape_names]

def numel(shape:list):
    p:int = 1
    for sz in shape:
        p *= sz
    return p

class SymbolicProduct:
    def __init__(self, names:list[str]):
        self.names = []
        self.const_factor = 1

        # expand names if needed
        idx_to_be_removed = []
        for i, n in enumerate(names):
            if '*' in n:
                # shape_name may contain *, e.g. "D*H", split it first to
                # ease further reductions
                idx_to_be_removed.append(i)
                names.extend([x.strip() for x in n.split('*')])
            elif n in g_var_registry and len(g_var_registry[n].reductions) > 0:
                idx_to_be_removed.append(i)
                names.extend(g_var_registry[n].reductions)
        for i in idx_to_be_removed[::-1]:
            del names[i]

        idx_to_be_removed.clear()
        name_count = len(names)
        for i in range(name_count):
            for j in range(i+1, name_count):
                n_s = names[i]
                n_m = names[j]
                if n_s > n_m:
                    n_s, n_m = n_m, n_s
                n2 = f"{n_s}*{n_m}"
                if n2 in g_merge_registry and len(g_merge_registry[n2].reductions) > 0:
                    #print(f"found n2={n2} i={i} j={j} names={names}")
                    idx_to_be_removed.append(i)
                    idx_to_be_removed.append(j)
                    e = g_merge_registry[n2]
                    names[i] = "" # must be erased
                    names[j] = "" 
                    names.extend(e.reductions)
                    break
        for i in idx_to_be_removed[::-1]:
            del names[i]
            
        for n in names:
            if is_num(n):
                self.const_factor *= float(n)
            else:
                for sub_var_name in n.split('*'):
                    sub_var_name = sub_var_name.strip()
                    if is_num(sub_var_name):
                        self.const_factor *= float(sub_var_name)
                    else:
                        self.names.append(sub_var_name)
            
        self.names.sort()

    def __repr__(self):
        desc = ""
        if self.const_factor != 1:
            desc += f"{self.const_factor}"
        last_name_single_alphabet = True
        for name in self.names:
            if len(desc) > 0 and (not last_name_single_alphabet):
                desc += "*"
            desc += name
            last_name_single_alphabet = len(name) == 1
        return desc

    def equal(self, other) -> bool:
        if len(self.names) != len(other.names):
            return False
        for i, n in enumerate(self.names):
            if n != other.names[i]:
                return False
        return True

class SymbolicExpr:
    def __init__(self):
        self.sum_of_prods: list[SymbolicProduct] = []

    def add_product(self, new_prod:SymbolicProduct):
        for prod in self.sum_of_prods:
            if prod.equal(new_prod):
                # merge with existing product
                prod.const_factor += new_prod.const_factor
                return 
        self.sum_of_prods.append(new_prod)

    def __repr__(self):
        return '+'.join([str(x) for x in self.sum_of_prods])

    def clear(self):
        self.sum_of_prods.clear()
        
class FakeTensor:
    accum_flops = 0
    accum_flops_expr = SymbolicExpr()
    print_flops_after_each_matmul = True

    @staticmethod
    def flops_str():
        return f"{FakeTensor.accum_flops_expr}={FakeTensor.accum_flops}"

    @staticmethod
    def print_flops():
        print(f"TotalFlops: {FakeTensor.flops_str()}")

    @staticmethod
    def clear_flops():
        FakeTensor.accum_flops = 0
        FakeTensor.accum_flops_expr.clear()

    def __init__(self, shapedesc: str|None):
        self.shape = []
        if shapedesc != None:
            self.shape_names = parse_shape_names(shapedesc)
            self.shape = get_shape_from_names(self.shape_names)
        else:
            self.shape_names = []
        assert len(self.shape) == len(self.shape_names)

    def clone(self):
        r = FakeTensor(None)
        r.shape_names = self.shape_names[:]
        r.shape = self.shape[:]
        return r

    def __repr__(self):
        desc = f"({','.join(self.shape_names)})"
        desc += f"-({','.join([str(s) for s in self.shape])})"
        return desc

    def axis_count(self):
        return len(self.shape)

    def transpose(self, axis1, axis2):
        if axis1 != axis2:
            r = self.clone()
            r.shape[axis1], r.shape[axis2] = self.shape[axis2], self.shape[axis1]
            r.shape_names[axis1], r.shape_names[axis2] = self.shape_names[axis2], self.shape_names[axis1]
            return r
        else:
            return self

    def squeeze_(self, dim: int|None = None):
        if dim is not None:
            if self.shape[dim] == 1:
                del self.shape[dim]
                del self.shape_names[dim]
            else:
                raise Exception(f"dim_{dim} is not 1, actually {self.shape[dim]}")
        else:
            new_shape = []
            for i in range(len(self.shape)):
                if self.shape[i] == 1:
                    del self.shape_names[i]
                else:
                    new_shape.append(self.shape[i])
            self.shape = new_shape
        return self

    def unsqueeze_(self, dim:int):
        self.shape.insert(dim, 1)
        self.shape_names.insert(dim, "1")
        return self
            
    def view(self, newshape_desc):
        newshape_names = parse_shape_names(newshape_desc)
        newshape = get_shape_from_names(newshape_names)
        assert len(newshape) != 0 and len(self.shape) != 0
        i1 = i2 = 0
        p1 = p2 = 0
        while i1 < len(newshape) and i2 < len(self.shape):
            p1 = newshape[i1]
            p2 = self.shape[i2]
            if p1 < p2:
                while p1 < p2 and i1 < len(newshape)-1:
                    i1 += 1
                    p1 *= newshape[i1]
            elif p2 < p1:
                while p2 < p1 and i2 < len(self.shape)-1:
                    i2 += 1
                    p2 *= self.shape[i2]
            if p1 != p2:
                raise Exception(f"Cannot view {self.shape} as {newshape}")
            i1 += 1
            i2 += 1
        while i1 < len(newshape):
            p1 *= self.shape[i1]
            i1 += 1
        while i2 < len(self.shape):
            p2 *= self.shape[i2]
            i2 += 1
        if p1 != p2:
            raise Exception(f"Cannot view {self.shape} as {newshape}")
        r = FakeTensor(None)
        r.shape_names = newshape_names
        r.shape = newshape
        return r
    
    # if `causal' is True, the calculated flops will be half-ed
    def matmul(self, other, causal = False):
        if not isinstance(other, FakeTensor):
            raise Exception("ERR: other must be FakeTensor as well")
        assert other.axis_count() >= 2, "RHS tensor must have at least 2 dims"
        assert self.axis_count() >= 2, "LHS tensor must have at least 2 dims"
        assert self.shape[-1] == other.shape[-2], f"dims for matmul does not match, ({self.shape[-2]},{self.shape[-1]})@({other.shape[-2]},{other.shape[-1]}) self={self} other={other}"
        min_axis = min(other.axis_count(), self.axis_count())
        # check broadcastable dims
        for axis in range(2, min_axis):
            sz1 = self.shape[-axis-1]
            sz2 = other.shape[-axis-1]
            if sz1 != sz2 and sz1 != 1 and sz2 != 1:
                raise Exception(f"Unmatched dim at [{-axis}], one of them must be 1 to follow broadcasting rules, self={self} other={other}")
        max_axis = max(other.axis_count(), self.axis_count())
        new_shape = [0] * max_axis
        def safe_get(l, idx):
            return l[idx] if idx < len(l) and idx >= 0 else 0
        ret = FakeTensor(None)
        ret.shape_names = ["-"] * len(new_shape)
        pad_self = max_axis - len(self.shape)
        pad_other = max_axis - len(other.shape)
        batch_factor = 1
        new_flops_prod_list = []
        for axis in range(0, max_axis - 2):
            d1 = safe_get(self.shape, axis - pad_self)
            d2 = safe_get(other.shape, axis - pad_other)
            new_d = max(d1, d2)
            new_shape[axis] = new_d
            new_name = self.shape_names[axis] if d1 > d2 else other.shape_names[axis]
            ret.shape_names[axis] = new_name
            batch_factor *= new_d
            new_flops_prod_list.append(new_name)
        new_shape[-2] = self.shape[-2]
        new_shape[-1] = other.shape[-1]
        ret.shape_names[-2] = self.shape_names[-2]
        ret.shape_names[-1] = other.shape_names[-1]
        ret.shape = new_shape
        new_flops = batch_factor * self.shape[-2] * self.shape[-1] * other.shape[-1]
        new_flops_prod_list.extend([self.shape_names[-2], self.shape_names[-1], other.shape_names[-1]])
        new_flops_expr = SymbolicProduct(new_flops_prod_list)
        if not causal:
            new_flops *= 2
            new_flops_expr.const_factor *= 2
        FakeTensor.accum_flops += new_flops
        assert new_flops_expr is not None
        FakeTensor.accum_flops_expr.add_product(new_flops_expr)
        if FakeTensor.print_flops_after_each_matmul:
            print(f"Matmul: {self} @ {other} -> {ret}, new_flops={new_flops_expr} total_flops={FakeTensor.flops_str()}")
        return ret

if __name__ == '__main__':
    register_variable("B", 1)
    register_variable("T", 4)
    register_variable("H", 2)
    register_variable("D", 4)
    register_variable("C", "D*H")
    # HACKY: a work-around to replace D*H with C to make the symbolic flops more readable
    register_variable("D*H", "C")

    print(f"g_var_registry: {g_var_registry}\ng_merge_registry: {g_merge_registry}");

    x = FakeTensor("B,T,C")
    Wq = FakeTensor("C,C")
    Wk = FakeTensor("C,C")
    Wv = FakeTensor("C,C")
    q = x.matmul(Wq).view("B,T,H,D").transpose(1,2)
    k = x.matmul(Wk).view("B,T,H,D").transpose(1,2)
    v = x.matmul(Wv).view("B,T,H,D").transpose(1,2)
    attn = q.matmul(k.transpose(-1,-2), causal=True)
    Wo = FakeTensor("C,C")
    merged_v = attn.matmul(v, causal=True).transpose(1,2).view("B,T,C")
    x2 = merged_v.matmul(Wo)
    fc1 = FakeTensor("C,4*C")
    fc2 = FakeTensor("4*C,C")
    x3 = x2.matmul(fc1).matmul(fc2)

    #print(x, Wq, Wk, Wv, q, k, v, attn, merged_v, x2, fc1, fc2, x3)
    FakeTensor.print_flops()

    print("\n=== clear ===")
    FakeTensor.clear_flops()
    register_variable("T", 1024)
    register_variable("H", 128)
    register_variable("D", 128)
    register_variable("C", 5120)
    register_variable("Dk", "0.1*C")
    register_variable("Dq", "0.3*C")
    # HACKY: a work-around to replace D*H with C to make the symbolic flops more readable
    register_variable("D*H", "3.2*C")
    register_variable("Tq", "T")

    print(f"g_var_registry: {g_var_registry}\ng_merge_registry: {g_merge_registry}");
    
    cur_x = FakeTensor("B,Tq,C")
    all_x = FakeTensor("B,T,C")
    W_dq = FakeTensor("C,Dq")
    W_kv = FakeTensor("C,Dk")
    C_q = cur_x.matmul(W_dq)
    C_kv = all_x.matmul(W_kv)
    W_uk = FakeTensor("Dk,D*H")
    W_uv = FakeTensor("Dk,D*H")
    k = C_kv.matmul(W_uk).view("B,T,H,D").transpose(1,2)
    v = C_kv.matmul(W_uv).view("B,T,H,D").transpose(1,2)
    W_uq = FakeTensor("Dq,D*H")
    q = C_q.matmul(W_uq).view("B,Tq,H,D").transpose(1,2)
    attn = q.matmul(k.transpose(-1,-2), causal=True)
    merged_v = attn.matmul(v, causal=True).transpose(1,2).view("B,Tq,D*H")
    Wo = FakeTensor("D*H,C")
    x2 = merged_v.matmul(Wo)

    FakeTensor.print_flops()
