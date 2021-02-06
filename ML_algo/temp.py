class Temp:
    def __init__(self, p):
        self.p = p
        pass

    def wrapper(func):
        def _func(self, arg):
            print(arg)
            print('wrapped')
            func(self, arg)
        return _func

    @wrapper
    def inClassFunc(self,arg):
        print('super: inClassFunc, value = ',self.p)
    def superCallsub(self,):
        self.inClassFunc(9)

class subTemp(Temp):
    def __init__(self, p):
        super().__init__(p)

    @Temp.wrapper
    def inClassFunc(self,arg):
        print('sub : inClassFunc, value = ', self.p)

if __name__ == '__main__':
    t = Temp(1)
    #t.inClassFunc(2)
    t = subTemp(2)
    t.inClassFunc(3)
    print()
    t.superCallsub()