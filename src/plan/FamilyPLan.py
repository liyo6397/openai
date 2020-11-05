import numpy as np

class Plan:

    def __init__(self, total_pay, user_data):

        self.total_pay = total_pay
        self.user_data = user_data
        self.total_data = np.sum(self.user_data)

    def base_cost(self):

        if self.total_data < 1:
            base_data = 0.25
            base_pay = 3
        else:
            base_data = 0.5
            base_pay = 6

        return base_data, base_pay

    def payment_base(self):


        user_pay = np.zeros_like(self.user_data)


        for i, data in enumerate(self.user_data):
            base_data = np.round(self.user_data)
            if base_data <= 1:
                user_pay[i] = 6
            else:
               user_pay[i] = 12*base_data

        return user_pay






