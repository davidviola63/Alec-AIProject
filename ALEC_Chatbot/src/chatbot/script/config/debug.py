# Flag di debug centralizzati
DEBUG_JUDGE = True
DEBUG_MODE  = True
DEBUG_CTX   = False
DEBUG_SAVE  = False

def dbg(msg: str):
    print(msg)

def dbg1(tag: str, value):
    print(f"DEBUG → {tag}: {repr(value)} (type={type(value)})")