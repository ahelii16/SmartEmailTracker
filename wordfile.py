
def func(k):
    words_list = {
        "ocean" : "collateral", 
        "le" : "legal entity"
    }

    if k in words_list.keys():
        return words_list[k]
    else :
        return None

# c = func("hi")
# print(c)