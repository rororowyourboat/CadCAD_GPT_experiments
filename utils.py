# plan parser function which takes a string and returns a list of functions to call. It uses the \n as a delimiter to split the string into a list of functions to call.
def plan_parser(plan):
    plan = plan.split('###')[1]
    plans = plan.split('\n')
    return plans


# pritn with colors
def print_color(string, color):
    print("\033["+color+"m"+string+"\033[0m")