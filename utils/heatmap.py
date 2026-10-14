import cv2



def create_heatmap(mask):


    heatmap = mask * 255


    heatmap = heatmap.astype(
        "uint8"
    )


    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )


    return heatmap