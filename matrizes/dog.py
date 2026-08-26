
#um método  é uma funão que faz parte de uma classe
#__init__() é um metodo especial que python executa automaticamente quando a gente cria uma nova instancia aseada na classe dog


class Dog():
    def __init__(self, name, age):
        self.name = name #variaveis acessadas por meio de instancias são so atributos
        self.age = age
        
    def sit(self): 
        print(self.name.title() + " is now sitting." )
    
    def roll_over(self):
        print(self.name.title() + "rolled over!")

#instancia para a ciração de um cachorro

my_dog = Dog('willie',6)
print("My dog's name is " + my_dog.name.title() + ".")

print("My dog is " + str(my_dog.age) + " years old.")

my_dog.sit()
my_dog.roll_over()