# Manually specify all attributes that we want to wrap
WRAPPED_METHODS = ['__add__', 'to_bytes']
WRAPPED_PROPERTIES = ['real']


def wrap_method(attr):
    def wrapped(self, *args, **kwargs):
        f = getattr(self.result, attr)
        return f(*args, **kwargs)
    return wrapped


def wrap_property(attr):
    def wrapped(self):
        return getattr(self.result, attr)
    return property(wrapped)


class Foo(object):
    def __init__(self, x):
        self.result = x


for method in WRAPPED_METHODS:
    setattr(Foo, method, wrap_method(method))
for prop in WRAPPED_PROPERTIES:
    setattr(Foo, prop, wrap_property(prop))

if __name__ == "__main__":
    a = Foo(100)
    print(a + 42)
    print(a.to_bytes(4, 'big'))
    print(a.real)
