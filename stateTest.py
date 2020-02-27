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

def makeCodesDropoff(maxR,maxC,maxP,maxD):
    codeSet = set()
    for r in range(maxR+1):
        for c in range(maxC+1):
            p = 4
            for d in range(maxD+1):
                if (r,c) not in {(0,0),(0,4),(4,0),(4,3)}:
                    codeSet.add(((((r*5)+c)*5+p)*4)+d)
    return codeSet

print("Dropoff")
print(makeCodesDropoff(4,4,4,3))
print(len(makeCodesDropoff(4,4,4,3)))

def makeCodesNorthWall(maxR, maxC, maxP, maxD):
    codeSet = set()
    for r in range(maxR + 1):
        for c in range(maxC + 1):
            for p in range(maxP + 1):
                for d in range(maxD + 1):
                    if (r, c) in {(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)}:
                        codeSet.add(((((r * 5) + c) * 5 + p) * 4) + d)
    return codeSet

print("North")
print(makeCodesNorthWall(4,4,4,3))
print(len(makeCodesNorthWall(4,4,4,3)))

def makeCodesSouthWall(maxR, maxC, maxP, maxD):
    codeSet = set()
    for r in range(maxR + 1):
        for c in range(maxC + 1):
            for p in range(maxP + 1):
                for d in range(maxD + 1):
                    if (r, c) in {(4, 0), (4, 1), (4, 2), (4, 3), (4, 4)}:
                        codeSet.add(((((r * 5) + c) * 5 + p) * 4) + d)
    return codeSet

print("South")
print(makeCodesSouthWall(4,4,4,3))
print(len(makeCodesSouthWall(4,4,4,3)))

def makeCodesWestWall(maxR, maxC, maxP, maxD):
    codeSet = set()
    for r in range(maxR + 1):
        for c in range(maxC + 1):
            for p in range(maxP + 1):
                for d in range(maxD + 1):
                    if (r, c) in {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (3, 1), (4, 1), (0, 2), (1, 2), (3, 3), (4, 3)}:
                        codeSet.add(((((r * 5) + c) * 5 + p) * 4) + d)
    return codeSet

print("West")
print(makeCodesWestWall(4,4,4,3))
print(len(makeCodesWestWall(4,4,4,3)))

def makeCodesEastWall(maxR, maxC, maxP, maxD):
    codeSet = set()
    for r in range(maxR + 1):
        for c in range(maxC + 1):
            for p in range(maxP + 1):
                for d in range(maxD + 1):
                    if (r, c) in {(0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (3, 0), (4, 0), (0, 1), (1, 1), (3, 2), (4, 2)}:
                        codeSet.add(((((r * 5) + c) * 5 + p) * 4) + d)
    return codeSet

print("East")
print(makeCodesEastWall(4,4,4,3))
print(len(makeCodesEastWall(4,4,4,3)))