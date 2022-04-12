#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import sys

## load old user defined modules
from OldModules.the_useful_library import *


## ###############################################################
## PREPARE WORKSPACE
##################################################################
os.system("clear") # clear terminal window


## ###############################################################
## FUNCTION: writing 'forcing_generator.inp'
##################################################################
def funcWriteForceGen(
        filepath_new_file, filepath_ref_file,
        des_velocity         = 5.0,
        des_k_driv           = 2.0,
        des_k_min            = 1.0,
        des_k_max            = 3.0,
        des_solweight        = 1.0,
        des_spectform        = 1.0,
        des_energy_coeff     = 5.0e-3,
        des_end_time         = 100,
        des_nsteps_per_teddy = 10,
        num_space_pad        = 37
    ):
    ## initialise boolean variables
    bool_set_velocity         = False
    bool_set_k_driv           = False
    bool_set_k_min            = False
    bool_set_k_max            = False
    bool_set_solweight        = False
    bool_set_spectform        = False
    bool_set_energy_coeff     = False
    bool_set_end_time         = False
    bool_set_nsteps_per_teddy = False
    ## initialise lines to write in file
    str_velocity         = "velocity                 = {}".format(des_velocity)
    str_k_driv           = "k_driv                   = {}".format(des_k_driv)
    str_k_min            = "k_min                    = {}".format(des_k_min)
    str_k_max            = "k_max                    = {}".format(des_k_max)
    str_solweight        = "st_solweight             = {}".format(des_solweight)
    str_spectform        = "st_spectform             = {}".format(des_spectform)
    str_energy_coeff     = "st_energy_coeff          = {}".format(des_energy_coeff)
    str_end_time         = "end_time                 = {}".format(des_end_time)
    str_nsteps_per_teddy = "nsteps_per_turnover_time = {}".format(des_nsteps_per_teddy)
    ## open new file
    with open(filepath_new_file, "w") as new_file:
        ## open refernce file
        with open(filepath_ref_file, "r") as ref_file_lines:
            ## loop over lines in reference file
            for ref_line_elems in ref_file_lines:
                ## split contents
                list_ref_line_elems = ref_line_elems.split()
                ## handle empty lines
                if len(list_ref_line_elems) == 0:
                    new_file.write("\n")
                    continue
                ## found row where 'velocity' is defined
                elif list_ref_line_elems[0] == "velocity":
                    new_file.write("{}{}! Target velocity dispersion\n".format(
                        str_velocity,
                        " " * ( num_space_pad - len(str_velocity) )
                    ))
                    bool_set_velocity = True
                ## found row where 'k_driv' is defined
                elif list_ref_line_elems[0] == "k_driv":
                    new_file.write("{}{}! Characteristic driving scale in units of 2pi / Lx.\n".format(
                        str_k_driv,
                        " " * ( num_space_pad - len(str_k_driv) )
                    ))
                    bool_set_k_driv = True
                ## found row where 'k_min' is defined
                elif list_ref_line_elems[0] == "k_min":
                    new_file.write("{}{}! Minimum driving wavnumber in units of 2pi / Lx\n".format(
                        str_k_min,
                        " " * ( num_space_pad - len(str_k_min) )
                    ))
                    bool_set_k_min = True
                ## found row where 'k_max' is defined
                elif list_ref_line_elems[0] == "k_max":
                    new_file.write("{}{}! Maximum driving wavnumber in units of 2pi / Lx\n".format(
                        str_k_max,
                        " " * ( num_space_pad - len(str_k_max) )
                    ))
                    bool_set_k_max = True
                ## found row where 'st_solweight' is defined
                elif list_ref_line_elems[0] == "st_solweight":
                    new_file.write("{}{}! 1.0: solenoidal driving, 0.0: compressive driving, 0.5: natural mixture\n".format(
                        str_solweight,
                        " " * ( num_space_pad - len(str_solweight) )
                    ))
                    bool_set_solweight = True
                ## found row where 'st_spectform' is defined
                elif list_ref_line_elems[0] == "st_spectform":
                    new_file.write("{}{}! 0: band, 1: paraboloid, 2: power law\n".format(
                        str_spectform,
                        " " * ( num_space_pad - len(str_spectform) )
                    ))
                    bool_set_spectform = True
                ## found row where 'st_energy_coeff' is defined
                elif list_ref_line_elems[0] == "st_energy_coeff":
                    new_file.write("{}{}! Used to adjust to target velocity; scales with (velocity/velocity_measured)^3.\n".format(
                        str_energy_coeff,
                        " " * ( num_space_pad - len(str_energy_coeff) )
                    ))
                    bool_set_energy_coeff = True
                ## found row where 'end_time' is defined
                elif list_ref_line_elems[0] == "end_time":
                    new_file.write("{}{}! End time of forcing sequence in units of turnover times\n".format(
                        str_end_time,
                        " " * ( num_space_pad - len(str_end_time) )
                    ))
                    bool_set_end_time = True
                ## found row where 'nsteps_per_turnover_time' is defined
                elif list_ref_line_elems[0] == "nsteps_per_turnover_time":
                    new_file.write("{}{}! number of forcing patterns per turnover time\n".format(
                        str_nsteps_per_teddy,
                        " " * ( num_space_pad - len(str_nsteps_per_teddy) )
                    ))
                    bool_set_nsteps_per_teddy = True
                ## found line where comment overflows
                elif (list_ref_line_elems[0] == "!") and ("*" not in list_ref_line_elems[1]):
                    new_file.write("{}! {}\n".format(
                        " " * num_space_pad,
                        " ".join(list_ref_line_elems[1:])
                    ))
                ## otherwise write line contents
                else: new_file.write(ref_line_elems)
    ## check that all parameters have been defined
    if ((bool_set_velocity         is not None) and
        (bool_set_k_driv           is not None) and
        (bool_set_k_min            is not None) and
        (bool_set_k_max            is not None) and
        (bool_set_solweight        is not None) and
        (bool_set_spectform        is not None) and
        (bool_set_energy_coeff     is not None) and
        (bool_set_end_time         is not None) and
        (bool_set_nsteps_per_teddy is not None)):
        ## indicate function executed successfully
        print("\t> 'forcing_generator.inp' has been successfully written.")
        return True
    else:
        print("\t> ERROR: 'forcing_generator.inp' failed to write correctly.")
        return False


## ###############################################################
## DEFINE MAIN PROGRAM
## ###############################################################
def main():
    ## define file details
    filename_ref  = "forcing_generator_ref.inp"
    filename_edit = "forcing_generator_edit.inp"
    filepath_file = "/Users/dukekriel/Documents/Studies/TurbulentDynamo/data/"
    ## write an edited version of the reference file
    funcWriteForceGen(
        filepath_new_file = filepath_file + filename_edit,
        filepath_ref_file = filepath_file + filename_ref
    )


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
    main()
    sys.exit()


## END OF PROGRAM