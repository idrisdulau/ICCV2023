import os
import sys
import cv2
import sys
import math
import tqdm
import numpy
import scipy
import skimage

def removeSmallPix(img, minPixAreaToKeep=4):
    imgCopy = numpy.copy(img) 
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(imgCopy.astype(numpy.uint8), connectivity=8)
    for i in range(1, len(stats)):
        if stats[i, cv2.CC_STAT_AREA] <= minPixAreaToKeep:
            imgCopy[labels == i] = 0
    return imgCopy

def getMutualShortestDistancePairs(firstPList, secondPList):
    distances = scipy.spatial.distance.cdist(firstPList, secondPList, "euclidean")
    maxInt = numpy.iinfo(numpy.int32).max
    distances[distances == 0] = maxInt
    minDistIdx = numpy.argmin(distances, axis=1)
    corList = tuple(zip(numpy.arange(len(minDistIdx)),minDistIdx))
    idxValList = [a for a in corList if any((a == b[::-1]) for b in corList)]
    uniquePairsList = [pair for i, pair in enumerate(idxValList) if pair[::-1] not in idxValList[:i]]
    endPToLink = []
    for e in uniquePairsList:
        endPToLink.append([tuple(firstPList[e[0]]),tuple(firstPList[e[1]])])
    return endPToLink

def getLinePixList(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    if x1 < x2:
        sx = 1
    else:
        sx = -1

    if y1 < y2:
        sy = 1
    else:
        sy = -1

    err = dx - dy

    linePixList = []
    while True:
        linePixList.append((x1, y1))

        if x1 == x2 and y1 == y2:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return linePixList

def checkBranchingConditions(pair, labels, stats, mainLabels, endPointsList, crossPointsList, statsSkeleton, skeleton):
    skeletonCopy = numpy.copy(skeleton)
    assert(len(pair) == 2)
    p1, p2 = pair

    #region does pair's elements are from a similar CC ?
    if labels[p1] == labels[p2]:
        return []
    #endregion
    
    # #region is path length bigger than skeleton branch length ?
    # pathLength = scipy.spatial.distance.pdist(pair, "euclidean")[0] 
    # minCCArea = min(statsSkeleton[labels[p1], cv2.CC_STAT_AREA],statsSkeleton[labels[p2], cv2.CC_STAT_AREA])
    # if pathLength > minCCArea:
    #     print(statsSkeleton[labels[p1], cv2.CC_STAT_AREA],statsSkeleton[labels[p2], cv2.CC_STAT_AREA])
    #     print("pathLength", pathLength, "minCCArea",minCCArea)
    #     print("path |",p1,p2)
    #     cv2.imshow("ig",skeletonCopy.astype(numpy.uint8)*255)
    #     cv2.waitKey(0)
    #     exit()
    #     return []
    # #endregion
    
    #region is the connection angle less than 135 degree ?    
    if labels[p1] != mainLabels and labels[p2] == mainLabels:
        smallestCCPix = p1
    elif labels[p1] == mainLabels and labels[p2] != mainLabels:
        smallestCCPix = p2
    #Size comparison done on img not on skeleton
    elif stats[labels[p1], cv2.CC_STAT_AREA] < stats[labels[p2], cv2.CC_STAT_AREA]: 
        smallestCCPix = p1
    else:
        smallestCCPix = p2

    branchPix = numpy.argwhere(labels==labels[smallestCCPix])
    branchEndP = [(a[0],a[1]) for a in branchPix if any((a == b).all() for b in endPointsList)]
    branchcrossP = [(a[0],a[1]) for a in branchPix if any((a == b).all() for b in crossPointsList)]
    branchEndCrossP = branchEndP+branchcrossP
       
    branchEndCrossP = [t for t in branchEndCrossP if t not in pair]
    distances = scipy.spatial.distance.cdist([smallestCCPix], branchEndCrossP, "euclidean")
    nearest = branchEndCrossP[numpy.argmin(distances)]
        
    if smallestCCPix == p1:
        uniquebranchEndCrossP = pair[::-1]+[nearest]
    else:
        uniquebranchEndCrossP = pair+[nearest]

    p1, p2, p3 = uniquebranchEndCrossP #Angle in p2
    assert(p1 != p2 and p2 != p3)
    a = numpy.array([p2[0] - p1[0], p2[1] - p1[1]])
    b = numpy.array([p2[0] - p3[0], p2[1] - p3[1]])

    # Clamp to avoid out of range cuz of precision loss
    arg = numpy.dot(a, b)/(numpy.linalg.norm(a)*numpy.linalg.norm(b))
    arg = numpy.clip(arg, -1, 1)
    angle = round(math.degrees(math.acos(arg)))

    #acos => angle€[0;180] 
    if angle < 135:
        return []
    #endregion

    #region is there already a vessel in the rebranching path ?
    linePixList = getLinePixList(p1, p2)
    linePixList.remove(p1)
    linePixList.remove(p2)

    skeletonCopy = skeletonCopy.astype(numpy.uint8)*255
    sumPath = 0
    for e in linePixList:
        sumPath += skeleton[e] #Does not work for diagonal crossing
    
    if sumPath > 0:
        return []
    #endregion

    return pair

def propagateBranchWidth(img, startP, kernelSize=2):
    y, x = startP
    imgCopy = numpy.copy(img)
    imgPadded = numpy.pad(imgCopy, pad_width=kernelSize, mode='constant', constant_values=0)
    y += kernelSize
    x += kernelSize
    pattern = imgPadded[x-kernelSize : x+kernelSize+1, y-kernelSize : y+kernelSize+1] 
    pattern = numpy.where(pattern == 0, pattern, 255)

    rows, cols = numpy.where(imgPadded == 200)
    for r, c in zip(rows, cols):
        rMin = max(r-kernelSize, 0)
        rMax = min(r+kernelSize+1, imgPadded.shape[0])
        cMin = max(c-kernelSize, 0)
        cMax = min(c+kernelSize+1, imgPadded.shape[1])

        mask = imgPadded[rMin:rMax, cMin:cMax] != 255
        imgPadded[rMin:rMax, cMin:cMax][mask] = pattern[mask]
    
    imgUnPadded = imgPadded[kernelSize:-kernelSize, kernelSize:-kernelSize]
    imgUnPadded = imgUnPadded.astype(numpy.uint8)
    return imgUnPadded
    
def gapFilling(img):
    numLabels, labels, stats, _ = cv2.connectedComponentsWithStats(img.astype(numpy.uint8), connectivity=8)
    mainLabels = numpy.argmax(stats[1:, cv2.CC_STAT_AREA])+1
    # mainPix = numpy.argwhere(labels == mainLabels)[:, ::-1]

    skeleton = skimage.morphology.skeletonize(img) 
    kernel = numpy.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    countNeighborsOfAllPix = scipy.signal.convolve2d(skeleton, kernel, mode='same', boundary='fill', fillvalue=0)
    countNeighborsOfOnesPix = numpy.where(skeleton==1, countNeighborsOfAllPix, 0)
    endPointsList = numpy.argwhere(countNeighborsOfOnesPix==1) #y,x format
    crossPointsList = numpy.argwhere(countNeighborsOfOnesPix==3) #y,x format

    _, labelsSkeleton, statsSkeleton, _ = cv2.connectedComponentsWithStats(skeleton.astype(numpy.uint8), connectivity=8)
    # mainPixSkeleton = numpy.argwhere(labelsSkeleton == mainLabels)[:, ::-1]
    # mainEndPixSkeleton = numpy.array([a for a in endPointsList if any((a == b).all() for b in mainPixSkeleton)])

    endPToLink = getMutualShortestDistancePairs(endPointsList, endPointsList)
    
    validPairs = []
    for e in endPToLink:
        pair = checkBranchingConditions(e,labels, stats, mainLabels, endPointsList, crossPointsList, statsSkeleton, skeleton)
        if pair!=[]:
            validPairs.append(pair) 
    # print("valid endPToLink:",len(validPairs),":", validPairs,"\n")

    crossEndPToLink = getMutualShortestDistancePairs(endPointsList, crossPointsList)

    for e in crossEndPToLink:
        pair = checkBranchingConditions(e,labels, stats, mainLabels, endPointsList, crossPointsList, statsSkeleton, skeleton)
        if pair!=[]:
            validPairs.append(pair) 
    # print("+ valid cross:",len(validPairs),":", validPairs)
    # print()

    # print(validPairs)
    # skeleton = skeleton.astype(numpy.uint8)*50
    for vp in validPairs:
        y0,x0 = vp[0]
        y1,x1 = vp[1]
        # cv2.line(skeleton, (x0,y0), (x1,y1) , 255, 1)
        cv2.line(img, (x0,y0), (x1,y1) , 200, 1)
        img = propagateBranchWidth(img, (x1,y1))
    # cv2.imshow("labels", labels.astype(numpy.uint8)*2)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    return img

def main(argv):
    inputPath, outputPath = argv[1:3]
    for imgName in tqdm.tqdm(os.listdir(inputPath)):
        img = cv2.imread(os.path.join(inputPath,imgName), cv2.IMREAD_UNCHANGED)
        img = removeSmallPix(img)
        img = gapFilling(img)
        cv2.imwrite(os.path.join(outputPath,imgName), img) 

if __name__ == '__main__':
    main(sys.argv)
