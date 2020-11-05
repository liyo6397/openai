import numpy as np
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()

class Plan:
    def __init__(self, user, total_fee = None):
        self.user = user
        self.person_fee = np.zeros_like(user)
        self.total_g = np.sum(self.user)
        self.total_fee = total_fee
        self.num_layer = int(np.round(self.total_g))


    def update_layer_data(self, data_usage, surplus):
        ave_data = (1 - np.sum(surplus)) / len(surplus)

        for i in range(len(data_usage)):
            if data_usage[i] > 0.25:
                data_usage[i] = ave_data

        return data_usage


    def layer_data(self):
        data_usage = np.zeros_like(self.user)
        surplus = []
        for i in range(len(self.user)):
            if self.user[i] > 0:
                if self.user[i] > 0.25:
                    self.user[i] -= 0.25
                    data_usage[i] = 0.25
                else:
                    data_usage[i] = self.user[i]
                    surplus.append(self.user[i])
                    self.user[i] = 0

        return data_usage, surplus

    def layer_fee(self, data_usage):

        N = len(self.user)
        sum_data = np.sum(data_usage)

        for i in range(N):

            if data_usage[i] > 0:
                ratio = data_usage[i] / sum_data
                if not self.total_fee:
                    if self. total_g < 3:
                        self.person_fee[i] += 12*ratio
                    elif self.total_g >= 3 and self.total_g < 6:
                        self.person_fee[i] += 10 * ratio
                else:
                    layer_bill = self.total_fee/self.num_layer
                    self.person_fee[i] += layer_bill * ratio




    def PayFee(self):



        for l in range(self.num_layer):

            data_usage, surplus = self.layer_data()

            if len(surplus) > 0:
                data_usage = self.update_layer_data(data_usage, surplus)

            self.layer_fee(data_usage)

        return self.person_fee


if __name__ == '__main__':

    user = [0.1, 0.3, 0.7, 0.4]
    #user = parse_arguments()
    plan = Plan(user)
    total_fee = plan.PayFee()

    print("Total Fee: ", total_fee)
