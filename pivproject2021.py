

import sys

# arguments:
#   pivproject2021.py <task> <path_to_template> <path_to_output_folder> <path_to_input_folder>


print("arguments: {}".format(sys.argv))
print("number of arguments: {}".format(len(sys.argv)))

if len(sys.argv) == 5:
    task = sys.argv[1]
    template_image = sys.argv[2]
    output_path = sys.argv[3]
    input_image_raw = sys.argv[4]

    print("task: {}".format(task))
    print("template_path: {}".format(template_image))
    print("output_path: {}".format(output_path))
    print("input_path: {}".format(input_image_raw))


 

elif len(sys.argv) == 6:
    print("this is for exercise 4")

else:
    print("ERROR: wrong number of arguments")
    sys.exit(1) 


