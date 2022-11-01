import os, json
os.system("clear")

def saveDict2Json(filepath_file, input_dict):
  ## if json-file already exists, then append dictionary
  if os.path.isfile(filepath_file):
    append2JsonFile(filepath_file, input_dict)
  ## create json-file with dictionary
  else: createJsonFile(filepath_file, input_dict)

def createJsonFile(filepath_file, dict2save):
  with open(filepath_file, "w") as fp:
    json.dump(dict2save, fp, sort_keys=True, indent=2)
  print("Saved json-file:", filepath_file)

def append2JsonFile(filepath_file, dict2append):
  ## read json-file into dict
  with open(filepath_file, "r") as fp_r:
    dict_old = json.load(fp_r)
  ## append extra contents to dict
  dict_old.update(dict2append)
  ## update (overwrite) json-file
  with open(filepath_file, "w") as fp_w:
    json.dump(dict_old, fp_w, sort_keys=True, indent=2)
  print("Updated json-file:", filepath_file)

def main():
  filename = f"demo.json"
  my_dict_1 = {
    "my_int"   : 1,
    "my_float" : 4.0,
    "my_str"   : "hi there mate"
  }
  saveDict2Json(filename, my_dict_1)
  my_dict_2 = {
    "my_list" : [ 1, 2, 3, 4 ],
    "my_str"  : "bye bye dude"
  }
  saveDict2Json(filename, my_dict_2)

if __name__ == "__main__":
  main()

## end of demo program