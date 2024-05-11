from dataclasses import dataclass

def numel(sizes:list):
    p:int = 1
    for sz in sizes:
        p *= sz
    return p

@dataclass
class VarMeta:
    value: float
    expansions: list[str]

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
    
def get_varmeta_from_desc(desc: str, always_have_expansions: bool):
    cf, remaining_names = get_atomic_names(desc, True)
    p = cf
    for name in remaining_names:
        if not name in g_var_registry:
            raise Exception(f"Unknown desc={name}, register it with register_variable")
        p *= g_var_registry[name].value
    expansions = []
    if cf == 1 and len(remaining_names) == 1 and not always_have_expansions:
        # keep expansions empty for regular variables
        pass
    else:
        expansions.append(str(cf))
        expansions.extend(remaining_names)
    return VarMeta(p, expansions)

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
            g_var_registry[var_name] = get_varmeta_from_desc(val, False)
        else:
            g_merge_registry[var_name] = get_varmeta_from_desc(val, True)
    else:
        raise Exception("Unsupported type of val")

def get_shape_from_names(shape_names: list[str]):
    return [get_int_from_desc(var_name) for var_name in shape_names]

class ProductOfNames:
    def __init__(self, names:list[str]):
        self.names = []
        self.const_factor = 1

        # expand names if needed
        idx_to_be_removed = []
        for i, n in enumerate(names):
            if '*' in n:
                # shape_name may contain *, e.g. "D*H", split it first to
                # ease further expansions
                idx_to_be_removed.append(i)
                names.extend([x.strip() for x in n.split('*')])
            elif n in g_var_registry and len(g_var_registry[n].expansions) > 0:
                idx_to_be_removed.append(i)
                names.extend(g_var_registry[n].expansions)
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
                if n2 in g_merge_registry and len(g_merge_registry[n2].expansions) > 0:
                    #print(f"found n2={n2} i={i} j={j} names={names}")
                    idx_to_be_removed.append(i)
                    idx_to_be_removed.append(j)
                    e = g_merge_registry[n2]
                    names[i] = "" # must be erased
                    names[j] = "" 
                    names.extend(e.expansions)
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
        #print(f"input={names} out={self}")

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

def combine_products(sum_of_prods: list[ProductOfNames], new_prod:ProductOfNames):
    for prod in sum_of_prods:
        if prod.equal(new_prod):
            prod.const_factor += new_prod.const_factor
            return 
    sum_of_prods.append(new_prod)
        
class FakeTensor:
    accum_flops = 0
    flops_desc = []
    print_flops_after_each_matmul = True

    @staticmethod
    def flops_str():
        return f"{'+'.join([str(x) for x in FakeTensor.flops_desc])}={FakeTensor.accum_flops}"

    @staticmethod
    def print_flops():
        print(f"TotalFlops: {FakeTensor.flops_str()}")

    @staticmethod
    def clear_flops():
        FakeTensor.accum_flops = 0
        FakeTensor.flops_desc = []

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
        n1 = numel(newshape)
        n2 = numel(self.shape)
        # FIXME: not strictly right, should be fixed soon
        if n1 != n2:
            raise Exception(f"Unmatched view {n1} {n2}")
        r = FakeTensor(None)
        r.shape_names = newshape_names
        r.shape = newshape
        return r
    
    def matmul(self, other, causal = False):
        if not isinstance(other, FakeTensor):
            raise Exception("ERR: other must be FakeTensor as well")
        assert other.axis_count() >= 2, "other tensor must have at least 2 dims"
        assert self.axis_count() >= 2, "self tensor must have at least 2 dims"
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
        new_flops_desc_list = []
        for axis in range(0, max_axis - 2):
            d1 = safe_get(self.shape, axis - pad_self)
            d2 = safe_get(other.shape, axis - pad_other)
            new_d = max(d1, d2)
            new_shape[axis] = new_d
            new_name = self.shape_names[axis] if d1 > d2 else other.shape_names[axis]
            ret.shape_names[axis] = new_name
            batch_factor *= new_d
            new_flops_desc_list.append(new_name)
        new_shape[-2] = self.shape[-2]
        new_shape[-1] = other.shape[-1]
        ret.shape_names[-2] = self.shape_names[-2]
        ret.shape_names[-1] = other.shape_names[-1]
        ret.shape = new_shape
        new_flops = batch_factor * self.shape[-2] * self.shape[-1] * other.shape[-1]
        new_flops_desc_list.extend([f"{self.shape_names[-2]}", f"{self.shape_names[-1]}", f"{other.shape_names[-1]}"])
        new_flops_desc = ProductOfNames(new_flops_desc_list)
        if not causal:
            new_flops *= 2
            new_flops_desc.const_factor *= 2
        FakeTensor.accum_flops += new_flops
        assert new_flops_desc is not None
        combine_products(FakeTensor.flops_desc, new_flops_desc)
        if FakeTensor.print_flops_after_each_matmul:
            print(f"Matmul: {self} @ {other} -> {ret}, new_flops={new_flops_desc} total_flops={FakeTensor.flops_str()}")
        return ret

if __name__ == '__main__':
    register_variable("B", 1)
    register_variable("T", 4)
    register_variable("H", 2)
    register_variable("D", 4)
    register_variable("C", "D*H")
    # HACKY: a work-around to replace D*H with C to make the symbolic flops more readable
    register_variable("D*H", "C")

    x = FakeTensor("B,T,C")
    Wq = FakeTensor("C,C")
    Wk = FakeTensor("C,C")
    Wv = FakeTensor("C,C")
    q = x.matmul(Wq).view("B,T,H,D").transpose(1,2)
    print(q)
    k = x.matmul(Wk).view("B,T,H,D").transpose(1,2)
    v = x.matmul(Wv).view("B,T,H,D").transpose(1,2)
    attn = q.matmul(k.transpose(-1,-2), causal=True)
    Wo = FakeTensor("C,C")
    merged_v = attn.matmul(v, causal=True).view("B,T,C")
    x2 = merged_v.matmul(Wo)
    fc1 = FakeTensor("C,4*C")
    fc2 = FakeTensor("4*C,C")
    x3 = x2.matmul(fc1).matmul(fc2)

    #print(x, Wq, Wk, Wv, q, k, v, attn, merged_v, x2, fc1, fc2, x3)
    FakeTensor.print_flops()

    print("\n=== clear ===")
    FakeTensor.clear_flops()
    register_variable("H", 128)
    register_variable("D", 128)
    register_variable("C", 5120)
    register_variable("Dk", "0.1*C")
    register_variable("Dq", "0.3*C")
    # HACKY: a work-around to replace D*H with C to make the symbolic flops more readable
    register_variable("D*H", "3.2*C")
    
    x = FakeTensor("B,T,C")
    W_dq = FakeTensor("C,Dq")
    W_kv = FakeTensor("C,Dk")
    C_q = x.matmul(W_dq)
    C_kv = x.matmul(W_kv)
    W_uk = FakeTensor("Dk,D*H")
    W_uv = FakeTensor("Dk,D*H")
    k = C_kv.matmul(W_uk).view("B,T,H,D").transpose(1,2)
    v = C_kv.matmul(W_uv).view("B,T,H,D").transpose(1,2)
    W_uq = FakeTensor("Dq,D*H")
    q = C_q.matmul(W_uq).view("B,T,H,D").transpose(1,2)
    attn = q.matmul(k.transpose(-1,-2), causal=True)
    merged_v = attn.matmul(v, causal=True).view("B,T,D*H")
    Wo = FakeTensor("D*H,C")
    x2 = merged_v.matmul(Wo)

    FakeTensor.print_flops()
