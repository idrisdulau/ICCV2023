import os
import sys
import cv2
import sys
import math
import tqdm
import numpy
import scipy
import skimage

def checkBranchingConditionsV2(pair, labels, branchEndP, branchcrossP, statsSkeleton):
    assert(len(pair) == 2)
    endP, mainP = pair
    # print(pair)
    assert(labels[endP] != labels[mainP])

    #region is path length bigger than skeleton branch length ?
    pathLength = scipy.spatial.distance.pdist(pair, "euclidean")[0] 
    minCCArea = min(statsSkeleton[labels[endP], cv2.CC_STAT_AREA],statsSkeleton[labels[mainP], cv2.CC_STAT_AREA])
    # print(pathLength , minCCArea)
    if pathLength > minCCArea:
        return False
    #endregion
    
    #region is the connection angle less than 135 degree ?    
    branchEndCrossP = branchEndP+branchcrossP
    branchEndCrossP.remove(endP)
    
    distances = scipy.spatial.distance.cdist([endP], branchEndCrossP, "euclidean")
    nearestEndCrossP = branchEndCrossP[numpy.argmin(distances)]

    p1, p2, p3 = mainP, endP, nearestEndCrossP #Angle in p2
    assert(p1 != p2 and p2 != p3)
    a = numpy.array([p2[0] - p1[0], p2[1] - p1[1]])
    b = numpy.array([p2[0] - p3[0], p2[1] - p3[1]])

    # Clamp to avoid out of range cuz of precision loss#
    arg = numpy.dot(a, b)/(numpy.linalg.norm(a)*numpy.linalg.norm(b))
    arg = numpy.clip(arg, -1, 1)
    angle = round(math.degrees(math.acos(arg)))

    # print(angle)
    #acos => angle€[0;180] 
    if angle < 135:
        return False
    #endregion

    return True

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

def getPListBelongsToLine(img, nearestEndCrossP, endP):
    x1, y1 = nearestEndCrossP
    x2, y2 = endP
    
    # Bresenham's line algorithm
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = -1 if x1 > x2 else 1
    sy = -1 if y1 > y2 else 1
    err = dx - dy
    
    linePList = []
    while True:
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x2 += sx
        if e2 < dx:
            err += dx
            y2 += sy
        
        if x2 < 0 or x2 > img.shape[0]-1 or y2 < 0 or y2 > img.shape[1]-1:
            break

        linePList.append((x2, y2))
    return linePList

def rebranchOrRemove(img):
    imgCopy = numpy.copy(img)
    numLabels, labels, stats, _ = cv2.connectedComponentsWithStats(img.astype(numpy.uint8), connectivity=8)
    mainLabels = numpy.argmax(stats[1:, cv2.CC_STAT_AREA])+1
    mainPix = numpy.argwhere(labels == mainLabels)

    skeleton = skimage.morphology.skeletonize(img) 
    skeletonCopy = numpy.copy(skeleton) 
    skeletonCopy = skeletonCopy.astype(numpy.uint8)*255
    kernel = numpy.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    countNeighborsOfAllPix = scipy.signal.convolve2d(skeleton, kernel, mode='same', boundary='fill', fillvalue=0)
    countNeighborsOfOnesPix = numpy.where(skeleton==1, countNeighborsOfAllPix, 0)
    endPointsList = numpy.argwhere(countNeighborsOfOnesPix==1) #y,x format
    crossPointsList = numpy.argwhere(countNeighborsOfOnesPix==3) #y,x format

    _, labelsSkeleton, statsSkeleton, _ = cv2.connectedComponentsWithStats(skeleton.astype(numpy.uint8), connectivity=8)
    mainPixSkeleton = numpy.argwhere(labelsSkeleton == mainLabels)
    # mainEndPixSkeleton = numpy.array([a for a in endPointsList if any((a == b).all() for b in mainPixSkeleton)])
    
    toRebranchList = []
    toRemoveList = []
    for label in range(1, numLabels):
        if label == mainLabels:
            continue

        branchPix = numpy.argwhere(labels == label) 
        branchEndP = [(a[0],a[1]) for a in branchPix if any((a == b).all() for b in endPointsList)]
        branchcrossP = [(a[0],a[1]) for a in branchPix if any((a == b).all() for b in crossPointsList)]

        # corList = []
        rebranchList = []
        for endP in branchEndP:

            # print(branchEndP, branchcrossP)

            branchEndCrossP = branchEndP+branchcrossP
            branchEndCrossP.remove(endP)
            distances = scipy.spatial.distance.cdist([endP], branchEndCrossP, "euclidean")
            nearestEndCrossP = branchEndCrossP[numpy.argmin(distances)]

            pList = getPListBelongsToLine(img, nearestEndCrossP, endP)

            for p in pList:
                if img[p] == 255 and labels[p] == mainLabels:
                    pathLength = scipy.spatial.distance.cdist([endP], [p], "euclidean")
                    rebranchList.append((endP,p,pathLength[0][0]))
                    break
        
        status = "removed"
        sortedrebranchList = sorted(rebranchList, key=lambda x: x[2]) #rebranch the first valid in SHORTEST order
        # print(sortedrebranchList)

        for endP, newP, pathLength in sortedrebranchList:
            
            if rebranchList != []:
                minCCArea = statsSkeleton[labelsSkeleton[endP], cv2.CC_STAT_AREA]  
                # print(endP)     
                # print(pathLength, minCCArea, endP, newP)
                if (pathLength < minCCArea) and (status != "rebranched"):
                    status = "rebranched"
                    toRebranchList.append((endP, newP))
                    # cv2.line(img, endP[::-1], newP[::-1], 200, 1)
                    # cv2.line(skeletonCopy, endP[::-1], newP[::-1], 200, 1)
                    # img = propagateBranchWidth(img, endP[::-1])
                    # print(pathLength, minCCArea)
                    # print("rebranched")
                    break
            
        if status == "removed":
            toRemoveList.append(branchPix)
            # for (y,x) in zip(branchPix[:,0], branchPix[:,1]):
            #     img[y, x] = 0
                # skeletonCopy[y,x] = 50
            
    for b in toRemoveList:
        for (y,x) in zip(b[:,0], b[:,1]):
            img[y, x] = 0

    for endP, newP in toRebranchList:
        cv2.line(img, endP[::-1], newP[::-1], 200, 1)
        img = propagateBranchWidth(img, endP[::-1])     
    
    # cv2.imshow("final",img)
    # cv2.waitKey(0)
    # exit()
    return img

def main(argv):
    inputPath, outputPath = argv[1:3]
    for imgName in tqdm.tqdm(os.listdir(inputPath)):
        img = cv2.imread(os.path.join(inputPath,imgName), cv2.IMREAD_UNCHANGED)
        img = rebranchOrRemove(img)
        cv2.imwrite(os.path.join(outputPath,imgName), img) 

if __name__ == '__main__':
    main(sys.argv)
