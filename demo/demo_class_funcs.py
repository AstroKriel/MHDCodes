class MyClass():
  def __init__(self):
    self.a = 10
  
  def func(self, a):
    print(a)
  
  def func(self, a, b):
    print(a, b)
  
  def func(self, a, b, c):
    print(a, b, c)

def main():
  obj = MyClass()
  obj.func("1")
  obj.func("1", 2.0)
  obj.func("1", 2.0, 3)

if __name__ == "__main__":
  main()
