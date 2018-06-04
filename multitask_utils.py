def print_value(value, key):
    value_type = type(value)
    if value_type is dict:
        print_dict_types(value)
    elif value_type is list:
        print_list_types(value)
    else:
        print([(key, value_type)])

def print_dict_types(dict_to_print):
    for key in dict_to_print.keys():
        value = dict_to_print[key]
        print_value(value, key)

def print_list_types(list_to_print: list):
    for value in list_to_print:
        print_value(value, "in list")

