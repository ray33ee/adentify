import cv2 as cv
import numpy as np
import sys

import pytesseract
import os

# Takes an ad image and tries to locate the 'X' button which closes the app.
# Returns a dictionary containing the original image, processed image and information on the
# bounding rectangle.
# An array of sizes can be supplied to start with small sections and expanding to larger ones if searches fail
def image_to_x_rectangle(original, threshold=210, sizes=[50, 100, 200]):

    results = {}

    results["original"] = original # cv.imread(os.path.join(directory, filename))

    # Greyscale
    gray = cv.cvtColor(results["original"], cv.COLOR_BGR2GRAY)

    # Filter out low-contrast values
    ret, processed = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)

    width = processed.shape[1]

    for size in sizes:

        # Crop the image to the required size
        processed_image = processed[0: size, width - size: width]

        # Get all artifacts in the corner
        contours, hierarchy = cv.findContours(processed_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        binary = np.zeros((size, size, 1), np.uint8)

        results["artifacts"] = cv.drawContours(binary, contours, -1, (255, 255, 255), thickness=cv.FILLED)

        binary = np.zeros((size, size, 1), np.uint8)

        # Iterate over all artifacts and see if any are an 'X'
        for index in range(contours.__len__()):
            cv.drawContours(binary, contours, index, (255, 255, 255), thickness=cv.FILLED)
            cv.waitKey(0)

            colour = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)

            # cv.imshow(name + str(index), colour)

            # Feed the image into tesseract and get the dict of bounding boxes
            boxes = pytesseract.image_to_boxes(colour, config="--psm 10", output_type=pytesseract.Output.DICT)

            if boxes != {}:
                for i in range(len(boxes['char'])):
                    (text, x1, y2, x2, y1) = (
                    boxes['char'][i], boxes['left'][i], boxes['top'][i], boxes['right'][i], boxes['bottom'][i])
                    if text == "x" or text == "X" or text == "*":
                        # Convert the coordinates from local (with respect to the section)
                        # to global (with respect to the original image)
                        g_x1, g_y2, g_x2, g_y1 = (width - size + x1, size - y2, width - size + x2, size - y1)

                        results["local"] = {
                            "rectangle": ((x1, size - y1), (x2, size - y2)),
                            "mid": (int(x1 + (x2 - x1) / 2), int(size - y1 + (y1 - y2) / 2))
                        }

                        results["global"] = {
                            "rectangle": ((g_x1, g_y1), (g_x2, g_y2)),
                            "mid": (int(g_x1 + (g_x2 - g_x1) / 2), int(g_y1 + (g_y2 - g_y1) / 2))
                        }

                        results["size"] = size

                        results["status"] = "success"


                        return results

            binary.fill(0)

    results["status"] = "fail"
    return results


# Convert the results from image_to_x_rectangle into a dictionary that can be converted to and from json format
def format_results(results, resize_to=500):
    if results["status"] == "success":
        original = results["original"]

        width = original.shape[1]
        height = original.shape[0]

        tl, br = results["global"]["rectangle"]
        mid = results["global"]["mid"]

        l_tl, l_br = results["local"]["rectangle"]
        l_mid = results["local"]["mid"]

        resized = cv.resize(original, (resize_to, resize_to))

        scaled_mid = (int(mid[0] / width * resize_to), int(mid[1] / height * resize_to))
        scaled_tl = (int(tl[0] / width * resize_to), int(tl[1] / height * resize_to))
        scaled_br = (int(br[0] / width * resize_to), int(br[1] / height * resize_to))

        cv.line(resized, scaled_mid, scaled_mid, (0, 0, 255), 2)
        cv.rectangle(resized, scaled_tl, scaled_br, (0, 255, 0), 1)

        cv.rectangle(original, tl, br, (0, 255, 0), 1)
        cv.line(original, mid, mid, (0, 0, 255), 3)

        cropped = original[0: results["size"], width - results["size"]: width]

        artifacts = cv.cvtColor(results["artifacts"], cv.COLOR_GRAY2BGR)

        cv.rectangle(artifacts, l_tl, l_br, (0, 255, 0), 1)
        cv.line(artifacts, l_mid, l_mid, (0, 0, 255), 3)

        # cv.imshow('resize', resized)
        # cv.waitKey(0)

        return {
            "box": results["global"]["rectangle"],
            "point": results["global"]["mid"],
            "cropped": cropped,
            "resized": resized,
            "artifacts": artifacts,
        }
    else:
        return {}


def adentify(image, threshold=210, sizes=[50, 100, 200], resize_to=500):
    return format_results(image_to_x_rectangle(image, threshold, sizes), resize_to)


def test():
    for filename in os.listdir("E:\\Software Projects\\Python\\adentify\\ads"):
        if filename.endswith(".png"):
            res = format_results(image_to_x_rectangle(cv.imread(os.path.join("E:\\Software Projects\\Python\\adentify\\ads", filename)), sizes=[200]))

            print(filename)

            cv.imshow('cropped', res["resized"])
            cv.waitKey(0)


def debug(path):
    image = cv.imread(path)
    res = format_results(image_to_x_rectangle(image, sizes=[200]))

    cv.imshow('cropped', res["cropped"])
    cv.imshow('resized', res["resized"])
    cv.imshow('artifacts', res["artifacts"])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
        image = cv.imread(path)
        print(format_results(image_to_x_rectangle(image)))
    else:
        print("Invalid number of command line arguments")

