class Resaurant():
    def __init__(self,restaurant_name,cuisine_type):

        self.restaurant_name = restaurant_name
        self.cuisine_type = cuisine_type

    def describe_restaurant(self):

        print(self.restaurant_name.title())
        print(self.cuisine_type.title())

    def open_restaurant (self):

        print("O restaurante esta aberto")

restaurant = Resaurant('Spoleto','massa')

print("O nome do restaurante é " + restaurant.restaurant_name.title())
print("O tipo de cozinha do " + restaurant.restaurant_name.title() + " é " + restaurant.cuisine_type.title())

restaurant.describe_restaurant()
restaurant.open_restaurant()

#9.2

bandeco = Resaurant('bandejão','saudavel')
mcdonald = Resaurant('McDonald','Fast-Food')
bar = Resaurant('MAta-rato','petisco')

bandeco.describe_restaurant()
mcdonald.describe_restaurant()
bar.describe_restaurant()




        