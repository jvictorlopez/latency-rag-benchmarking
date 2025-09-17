#!/usr/bin/env python3
import os, sys, inspect

# Reuse API client creation
sys.path.append(os.path.abspath("api"))
from src.rag.weav_client import get_client, get_collection

def main():
    client = get_client()
    try:
        col = get_collection(client)
        q = col.query
        sig = None
        doc = ""
        if hasattr(q, "near_vector"):
            func = q.near_vector
            sig = str(inspect.signature(func))
            doc = (func.__doc__ or "")[:400]
        print("near helpers:", [n for n in dir(q) if n.startswith("near")])
        print("near_vector signature:", sig)
        print("near_vector doc:", doc)
    finally:
        client.close()

if __name__ == "__main__":
    main()


