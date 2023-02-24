from datetime import datetime

class Logger():

    def log_1(txt):
        print(f"{datetime.now()} ##### {txt} #####")

    def log_2(txt):
        print(f"{datetime.now()} # {txt} #")

    def log_3(txt):
        print(f"-/{datetime.now()}{txt}/-")
