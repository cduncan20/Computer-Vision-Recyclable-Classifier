
def interface():
    print("Data augmentation interface")

    print("Data Augmentation Manager")
    print("---")
    print("Select the image transformations you would like to add to the data. You may select as many as you like")
    print("Flip horizontally [1], Flip vertically [2], Flip vertically and horizontally [3], Rotate 45 degrees [4], "
          "Add artificial noise [5], Blur image [6], All [7] ")

    # Note to future tyler -> input is verbatim. 
    selected_augs = input()

    print(selected_augs)
