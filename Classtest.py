
class A(object):
    def __init__(self):
        self.attr1=1
        self.attr2=2

    @property
    def attr3(self) :
        if not hasattr(self, "attr3"):
            setattr(self, 'attr3', 18)
        return self.attr3
    def __call__(self, *args, **kwargs):
        print(*args)
        print("wow")
a=A()
print(a(':'))