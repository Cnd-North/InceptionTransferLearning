# Gerrit van Rensburg
# 2022-04-17

# this is to shuffle and move files into the validation set and the remainder will by moved into th etraining set

# current ~42,000 images were generated & 20% (8400) will be used for validation

ls | shuf -n 8400 | xargs -i mv {} path-to-new-folder

