from datetime import datetime

class Log():

    def log0(txt):
        print(f"{datetime.now()} ##### {txt} #####")

    def log1(txt):
        print(f"{datetime.now()} # {txt} #")

    def log3(txt):
        print(f"-/{datetime.now()}{txt}/-")
