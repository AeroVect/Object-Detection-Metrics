"""
Our modification still computes mAP the same way as Pascal VOC, but we added functionality to obtain class-specific optimal score threshold using F1 score.
For each score threshold, the Precision and Recall needed to compute the F1 score are not computed per image, but are instead computed using the total TP, FP, and FN for the class.
This way, we don't bias an "easier" image that contains less number of groundtruths (GT) than a "harder" image with more GT.

TLDR: Use COCO metrics for more robust mAP evaluation, but use this script to get the optimal score threshold for each class using F1 score.
"""

import argparse
import glob
import os
import shutil
import sys
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import _init_paths
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
from utils import BBFormat


# Validate formats
def ValidateFormats(argFormat, argName, errors):
    if argFormat == "xywh":
        return BBFormat.XYWH
    elif argFormat == "xyrb":
        return BBFormat.XYX2Y2
    elif argFormat is None:
        return BBFormat.XYWH  # default when nothing is passed
    else:
        errors.append(
            "argument %s: invalid value. It must be either 'xywh' or 'xyrb'" % argName
        )


# Validate mandatory args
def ValidateMandatoryArgs(arg, argName, errors):
    if arg is None:
        errors.append("argument %s: required argument" % argName)
    else:
        return True


def ValidateImageSize(arg, argName, argInformed, errors):
    errorMsg = "argument %s: required argument if %s is relative" % (
        argName,
        argInformed,
    )
    ret = None
    if arg is None:
        errors.append(errorMsg)
    else:
        arg = arg.replace("(", "").replace(")", "")
        args = arg.split(",")
        if len(args) != 2:
            errors.append(
                "%s. It must be in the format 'width,height' (e.g. '600,400')"
                % errorMsg
            )
        else:
            if not args[0].isdigit() or not args[1].isdigit():
                errors.append(
                    "%s. It must be in INdiaTEGER the format 'width,height' (e.g. '600,400')"
                    % errorMsg
                )
            else:
                ret = (int(args[0]), int(args[1]))
    return ret


# Validate coordinate types
def ValidateCoordinatesTypes(arg, argName, errors):
    if arg == "abs":
        return CoordinatesType.Absolute
    elif arg == "rel":
        return CoordinatesType.Relative
    elif arg is None:
        return CoordinatesType.Absolute  # default when nothing is passed
    errors.append(
        "argument %s: invalid value. It must be either 'rel' or 'abs'" % argName
    )


def ValidatePaths(arg, nameArg, errors):
    if arg is None:
        errors.append("argument %s: invalid directory" % nameArg)
    elif (
        os.path.isdir(arg) is False
        and os.path.isdir(os.path.join(currentPath, arg)) is False
    ):
        errors.append("argument %s: directory does not exist '%s'" % (nameArg, arg))
    # elif os.path.isdir(os.path.join(currentPath, arg)) is True:
    #     arg = os.path.join(currentPath, arg)
    else:
        arg = os.path.join(currentPath, arg)
    return arg


def getBoundingBoxes(
    directory,
    isGT,
    bbFormat,
    coordType,
    allBoundingBoxes=None,
    allClasses=None,
    imgSize=(0, 0),
):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    if allBoundingBoxes is None:
        allBoundingBoxes = BoundingBoxes()
    if allClasses is None:
        allClasses = []
    # Read ground truths
    os.chdir(directory)
    files = glob.glob("*.txt")
    files.sort()
    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box
    # (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in files:
        nameOfImage = f.replace(".txt", "")
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(" ", "") == "":
                continue
            splitLine = line.split(" ")
            if isGT:
                # idClass = int(splitLine[0]) #class
                idClass = splitLine[0]  # class
                x = float(splitLine[1])
                y = float(splitLine[2])
                w = float(splitLine[3])
                h = float(splitLine[4])
                bb = BoundingBox(
                    nameOfImage,
                    idClass,
                    x,
                    y,
                    w,
                    h,
                    coordType,
                    imgSize,
                    BBType.GroundTruth,
                    format=bbFormat,
                )
            else:
                # idClass = int(splitLine[0]) #class
                idClass = splitLine[0]  # class
                confidence = float(splitLine[1])
                x = float(splitLine[2])
                y = float(splitLine[3])
                w = float(splitLine[4])
                h = float(splitLine[5])
                bb = BoundingBox(
                    nameOfImage,
                    idClass,
                    x,
                    y,
                    w,
                    h,
                    coordType,
                    imgSize,
                    BBType.Detected,
                    confidence,
                    format=bbFormat,
                )
            allBoundingBoxes.addBoundingBox(bb)
            if idClass not in allClasses:
                allClasses.append(idClass)
        fh1.close()
    return allBoundingBoxes, allClasses


# Get current path to set default folders
currentPath = os.path.dirname(os.path.abspath(__file__))


parser = argparse.ArgumentParser(
    prog="Object Detection Metrics - modified for AeroVect's use",
)
parser.add_argument(
    "-gt",
    "--gtfolder",
    dest="gtFolder",
    default=os.path.join(currentPath, "groundtruths"),
    metavar="",
    help="folder containing your ground truth bounding boxes",
)
parser.add_argument(
    "-det",
    "--detfolder",
    dest="detFolder",
    default=os.path.join(currentPath, "detections"),
    metavar="",
    help="folder containing your detected bounding boxes",
)
parser.add_argument(
    "-t",
    "--threshold",
    dest="iouThreshold",
    type=float,
    default=0.5,
    metavar="",
    help="IOU threshold. Default 0.5",
)
parser.add_argument(
    "-gtformat",
    dest="gtFormat",
    metavar="",
    default="xywh",
    help="format of the coordinates of the ground truth bounding boxes: "
    "('xywh': <left> <top> <width> <height>)"
    " or ('xyrb': <left> <top> <right> <bottom>)",
)
parser.add_argument(
    "-detformat",
    dest="detFormat",
    metavar="",
    default="xywh",
    help="format of the coordinates of the detected bounding boxes "
    "('xywh': <left> <top> <width> <height>) "
    "or ('xyrb': <left> <top> <right> <bottom>)",
)
parser.add_argument(
    "-gtcoords",
    dest="gtCoordinates",
    default="abs",
    metavar="",
    help="reference of the ground truth bounding box coordinates: absolute "
    "values ('abs') or relative to its image size ('rel')",
)
parser.add_argument(
    "-detcoords",
    default="abs",
    dest="detCoordinates",
    metavar="",
    help="reference of the ground truth bounding box coordinates: "
    "absolute values ('abs') or relative to its image size ('rel')",
)
parser.add_argument(
    "-imgsize",
    dest="imgSize",
    metavar="",
    help="image size. Required if -gtcoords or -detcoords are 'rel'",
)
parser.add_argument(
    "-sp",
    "--savepath",
    dest="savePath",
    metavar="",
    help="folder where the plots are saved",
)
parser.add_argument(
    "-np",
    "--noplot",
    dest="showPlot",
    action="store_false",
    help="no plot is shown during execution",
)
args = parser.parse_args()

iouThreshold = args.iouThreshold

# Arguments validation
errors = []
# Validate formats
gtFormat = ValidateFormats(args.gtFormat, "-gtformat", errors)
detFormat = ValidateFormats(args.detFormat, "-detformat", errors)
# Groundtruth folder
if ValidateMandatoryArgs(args.gtFolder, "-gt/--gtfolder", errors):
    gtFolder = ValidatePaths(args.gtFolder, "-gt/--gtfolder", errors)
else:
    # errors.pop()
    gtFolder = os.path.join(currentPath, "groundtruths")
    if os.path.isdir(gtFolder) is False:
        errors.append("folder %s not found" % gtFolder)
# Coordinates types
gtCoordType = ValidateCoordinatesTypes(args.gtCoordinates, "-gtCoordinates", errors)
detCoordType = ValidateCoordinatesTypes(args.detCoordinates, "-detCoordinates", errors)
imgSize = (0, 0)
if gtCoordType == CoordinatesType.Relative:  # Image size is required
    imgSize = ValidateImageSize(args.imgSize, "-imgsize", "-gtCoordinates", errors)
if detCoordType == CoordinatesType.Relative:  # Image size is required
    imgSize = ValidateImageSize(args.imgSize, "-imgsize", "-detCoordinates", errors)
# Detection folder
if ValidateMandatoryArgs(args.detFolder, "-det/--detfolder", errors):
    detFolder = ValidatePaths(args.detFolder, "-det/--detfolder", errors)
else:
    # errors.pop()
    detFolder = os.path.join(currentPath, "detections")
    if os.path.isdir(detFolder) is False:
        errors.append("folder %s not found" % detFolder)
if args.savePath is not None:
    savePath = ValidatePaths(args.savePath, "-sp/--savepath", errors)
else:
    savePath = os.path.join(currentPath, "results")
# Validate savePath
# If error, show error messages
if len(errors) != 0:
    print(
        """usage: Object Detection Metrics [-h] [-v] [-gt] [-det] [-t] [-gtformat]
                                [-detformat] [-save]"""
    )
    print("Object Detection Metrics: error(s): ")
    [print(e) for e in errors]
    sys.exit()

# Check if path to save results already exists and is not empty
if os.path.isdir(savePath) and os.listdir(savePath):
    key_pressed = ""
    while key_pressed.upper() not in ["Y", "N"]:
        print(f"Folder {savePath} already exists and may contain important results.\n")
        print(
            f"Enter 'Y' to continue. WARNING: THIS WILL REMOVE ALL THE CONTENTS OF THE FOLDER!"
        )
        print(f"Or enter 'N' to abort and choose another folder to save the results.")
        key_pressed = input("")

    if key_pressed.upper() == "N":
        print("Process canceled")
        sys.exit()

# Clear folder and save results
shutil.rmtree(savePath, ignore_errors=True)
os.makedirs(savePath)

# Show plot during execution
showPlot = args.showPlot

# Get groundtruth boxes
allBoundingBoxes, allClasses = getBoundingBoxes(
    gtFolder, True, gtFormat, gtCoordType, imgSize=imgSize
)
# Get detected boxes
allBoundingBoxes, allClasses = getBoundingBoxes(
    detFolder,
    False,
    detFormat,
    detCoordType,
    allBoundingBoxes,
    allClasses,
    imgSize=imgSize,
)
allClasses.sort()

evaluator = Evaluator()


scores = np.arange(0, 1, 0.01)
ap_per_class = {}
f1_per_class = defaultdict(list)
precision_per_class = defaultdict(list)
recall_per_class = defaultdict(list)

for scoreThreshold in tqdm(scores):
    # Plot Precision x Recall curve

    # Only plot PR curve for no score thresholding
    if scoreThreshold == 0.0:
        savePath = savePath
    else:
        savePath = None

    detections = evaluator.PlotPrecisionRecallCurve(
        allBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
        scoreThreshold=scoreThreshold,  # score threshold
        IOUThreshold=iouThreshold,  # IOU threshold
        method=MethodAveragePrecision.EveryPointInterpolation,
        showAP=True,  # Show Average Precision in the title of the plot
        showInterpolatedPrecision=True,  # Don't plot the interpolated precision curve
        savePath=savePath,
        showGraphic=showPlot,
    )

    # each detection is a class
    for metricsPerClass in detections:
        # Get metric values per each class
        cl = metricsPerClass["class"]
        ap = metricsPerClass["AP"]
        totalPositives = metricsPerClass["total positives"]
        total_TP = metricsPerClass["total TP"]
        total_FP = metricsPerClass["total FP"]

        if scoreThreshold == 0.0:
            ap_per_class[cl] = ap

        if totalPositives > 0:
            precision = (
                total_TP / (total_TP + total_FP) if total_TP + total_FP > 0 else 1.0
            )
            recall = total_TP / totalPositives
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if precision + recall > 0
                else 0
            )
            f1_per_class[cl].append(f1)
            precision_per_class[cl].append(precision)
            recall_per_class[cl].append(recall)

# Obtain obtain score threshold for each class
print(f"IoU threshold={iouThreshold}")
for cl in allClasses:
    f1_values = f1_per_class[cl]
    if len(f1_values):
        score_i = np.argmax(f1_values)
        score = scores[score_i]
        print(
            f"Class {cl} - score: {format(score, '.2f')}, f1: {format(f1_values[score_i], '.2f')}, AP: {format(ap_per_class[cl], '.2f')}"
        )

mAP = np.mean(list(ap_per_class.values()))
print(f"mAP: {format(mAP, '.2f')}")
