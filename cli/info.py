class app:
    name        = "parallelepipedonn.py"
    version     = "2.2.0.5"
    author      = "Merkouris Papamichail"
    institute_1 = "Institute of Computer Science, FORTH"
    institute_2 = "Computer Science Department, ---"
    year        = "2026"
    license     = "CC-BY-NC-SA 4.0"


class info_header:
    logo1       = "   +-------------------* ub"
    logo2       = "   | ParallelepipedoNN |"
    logo3       = "lb *-------------------+"
    author      = "author:"
    institute   = "institute:"
    version     = "version:"
    year        = "year:"
    license     = "license:"



def print_header():
    print("")
    print(info_header.logo1)
    print(info_header.logo2)
    print(info_header.logo3)
    print("")
    print(f"{info_header.author:<12}"    + app.author)
    print(f"{info_header.institute:<12}" + app.institute_1)
    #print(f"{'':<12}"                    + app.institute_2)
    print(f"{info_header.version:<12}"   + app.version)
    print(f"{info_header.year:<12}"      + app.year)
    print(f"{info_header.license:<12}"   + app.license)
    print("")
