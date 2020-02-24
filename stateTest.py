def encode(taxi_row, taxi_col, pass_loc, dest_idx):
    # (5) 5, 5, 4
    i = taxi_row
    i *= 5
    i += taxi_col
    i *= 5
    i += pass_loc
    i *= 4
    i += dest_idx
    return i

#print(encode(0,0,4,2))

def makeCodes(maxR,maxC,maxP,maxD):
    codeSet = set()
    for r in range(maxR+1):
        for c in range(maxC+1):
            p = 4
            for d in range(maxD+1):
                if (r,c) not in {(0,0),(0,4),(4,0),(4,3)}:
                    codeSet.add(((((r*5)+c)*5+p)*5)+d)
    return codeSet

print(makeCodes(4,4,4,3))
print(len(makeCodes(4,4,4,3)))