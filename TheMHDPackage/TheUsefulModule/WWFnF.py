## START OF MODULE


## ###############################################################
## DEPENDENCIES: REQUIRED MODULES
## ###############################################################
import os, re
import numpy as np


## ###############################################################
## WORKING WITH FILES / FOLDERS
## ###############################################################
def makeFilter(
        str_contains       = None,
        str_not_contains   = None,
        str_startswith     = None,
        str_endswith       = None,
        file_index_placing = None,
        file_start_index   = 0,
        file_end_index     = np.inf,
        str_split_by       = "_"
    ):
    """ makeFilter
        PURPOSE: Create a filter condition for files that look a particular way.
    """
    def meetsCondition(element):
        ## if str_contains specified, then look for condition
        if str_contains is not None: bool_contains = element.__contains__(str_contains)
        else: bool_contains = True # don't consider condition
        ## if str_not_contains specified, then look for condition
        if str_not_contains is not None: bool_not_contains = not(element.__contains__(str_not_contains))
        else: bool_not_contains = True # don't consider condition
        ## if str_startswith specified, then look for condition
        if str_startswith is not None: bool_startswith = element.startswith(str_startswith)
        else: bool_startswith = True # don't consider condition
        ## if str_endswith specified, then look for condition
        if str_endswith is not None: bool_endswith = element.endswith(str_endswith)
        else: bool_endswith = True # don't consider condition
        ## make sure that the file has the right name structure (i.e. check all conditions have been met)
        if (
                bool_contains and 
                bool_not_contains and 
                bool_startswith and 
                bool_endswith
            ):
            ## if the index range also needs to be checked
            if file_index_placing is not None:
                ## check that the file index falls within the specified range
                if len(element.split(str_split_by)) > abs(file_index_placing):
                    bool_time_after  = (
                        int(element.split(str_split_by)[file_index_placing]) >= file_start_index
                    )
                    bool_time_before = (
                        int(element.split(str_split_by)[file_index_placing]) <= file_end_index
                    )
                    ## if the file meets all the required conditions
                    if (bool_time_after and bool_time_before): return True
            ## otherwise, all specified conditions have been met
            else: return True
        ## otherwise, don't look at the file
        else: return False
    return meetsCondition

def getFilesFromFolder(
        folder_directory, 
        str_contains       = None,
        str_startswith     = None,
        str_endswith       = None,
        str_not_contains   = None,
        file_index_placing = None,
        file_start_index   = 0,
        file_end_index     = np.inf
    ):
    ''' getFilesFromFolder
        PURPOSE: Return the names of files that meet the required conditions in the specified folder.
    '''
    myFilter = makeFilter(
        str_contains,
        str_not_contains,
        str_startswith,
        str_endswith,
        file_index_placing,
        file_start_index,
        file_end_index
    )
    return list(filter(myFilter, sorted(os.listdir(folder_directory))))

def createFolder(folder_name, bool_hide_updates=False):
    """ createFolder
    PURPOSE: Create a folder if and only if it does not already exist.
    """
    if not(os.path.exists(folder_name)):
        os.makedirs(folder_name)
        if not(bool_hide_updates):
            print("SUCCESS: Folder created. \n\t" + folder_name + "\n")
    elif not(bool_hide_updates):
        print("WARNING: Folder already exists (folder not created). \n\t" + folder_name + "\n")

def createFilepath(folder_names):
    """ creatFilePath
    PURPOSE: Concatinate a list of folder names into a single string separated by '/'.
    """
    return re.sub( '/+', '/', "/".join([folder for folder in folder_names if folder != ""]) )

def createName(name_elems):
    """ creatFilePath
    PURPOSE: Concatinate a list of folder names into a single string separated by '_.
    """
    return re.sub( '_+', '_', "_".join([elems for elems in name_elems if elems != ""]) )


## END OF MODULE