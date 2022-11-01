## START OF LIBRARY


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def assertType(var_name, var, req_types):
  if not isinstance(var, req_types):
    raise Exception(f"ERROR: Variable '{var_name}' type is type '{type(var)}' instead of '{req_types}'")


## END OF LIBRARY