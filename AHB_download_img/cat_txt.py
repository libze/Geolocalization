import os 

top_folder = "./AHB_download_img/mp16"
for folder_letter in os.listdir(top_folder):
    for folder_country in os.listdir(top_folder + "/" + folder_letter):
        print(folder_country)