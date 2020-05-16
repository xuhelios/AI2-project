# Author: TRAN Ngoc Nhat Huyen + NGUYEN Phuong Hoa
# Created: April, 2020
# ____

from tkinter import *
from kmean import *
from logistic import *
from perceptron import *
from single_layer_neural_network import *

size = 150


class Project:

    def __init__(self):
        self.count = 0
        self.window = Tk()
        self.window.title('AI-2 Project')
        self.window.geometry("420x650")
        info = Label(self.window, text="CONTROL BOARD", font='Helvetica 15 bold',
                     justify=CENTER, padx=108, pady=15, fg="#0252A4").pack(side=TOP)
        self.frame = Frame(self.window, width=size, height=size/2)
        self.frame.pack(side=TOP)
        self.framek = Frame(self.frame, width=size/2, height=size/2)
        self.framek.pack(side=LEFT)
        self.btnFrame = Frame(self.frame, width=size/2, height=size/2)
        self.btnFrame.pack(side=LEFT)
        self.v = IntVar(self.window)
        self.v.set(0)

        self.data = self.gen_data(50)
        k_choice = [
            ("K=", 2),
            ("K=", 3),
            ("K=", 4)]

        Label(self.framek, text="""Choose K:""",
              justify=LEFT, padx=20).pack()

        for val, language in enumerate(k_choice):
            Radiobutton(self.framek,
                        text=language,
                        pady=10,
                        variable=self.v,
                        value=val).pack()
        Label(self.btnFrame, text="Unsupervised learning:", font='Helvetica 11 bold',
              justify=CENTER, pady=10).pack()
        self.submit = Button(self.btnFrame,  width=30, pady=10,
                             text="Solve K-means", command=self.question1)
        self.submit.pack()
        Label(self.btnFrame, text="Supervised learning:",font='Helvetica 11 bold',
              justify=LEFT, pady=10).pack()


        self.perceptron = Button(self.btnFrame, width=30, pady=10,
                           text="3.1 Perceptron & Logistic",  command= self.question2)
        self.perceptron.pack()

        self.logistic = Button(self.btnFrame, width=30, pady=10, text="3.2 Perceptron & Neural network", command = self.question3,
                            relief=GROOVE)
        self.logistic.pack()

        self.text = Text(self.window, width=size*3 , height=size*2.5, bg="#e6f2ff")
        self.text.pack()

        # self.dominated = Button(self.btnFrame, width=20, pady=5, text="Dominated Strategies",
        #                         command=self.dominated_strategies, relief=GROOVE)
        # self.dominated.pack()
        # self.text = Text(self.solFrame, width=50, height=10,
        #                  wrap=WORD, bg="lavender")
        # self.text.grid(row=0, column=0)
        # self.text.config(state="normal")

        # self.initialize_board()

    def read_training_data_type1(self, data):
        training_data = []
        training_class = []

        for i in range(len(data)):
            group = data[i][0]
            point_x = data[i][1]
            point_y = data[i][2]
            if int(group) < 2:
                label = 0
            else:
                label = 1

            training_data.append([float(point_x), float(point_y)])
            training_class.append(int(label))

        return training_data, training_class

    def read_training_data_type2(self, data):
        training_data = []
        training_class = []

        for i in range(len(data)):
            group = data[i][0]
            point_x = data[i][1]
            point_y = data[i][2]
            if group == 0 or group == 3: #(groupe 1 et groupe 4 => class 2)
                label = 1
            else:
                label = 0

            training_data.append([float(point_x), float(point_y)])
            training_class.append(int(label))

        return training_data, training_class

    def gen_test_data(self):
        test_data = []
        test_class = []
        group = np.random.uniform(low=0, high=1, size=(50, 2))

        for x, y in group:
            d = [1 + x, 1 + y]  # group1 -> class1
            test_data.append(d)
            test_class.append(0)

        for x, y in group:
            d = [1 + x, -2 + y]  # group2 -> class1
            test_data.append(d)
            test_class.append(0)

        for x, y in group:
            d = [-2 + x, 1 + y]  # group3 -> class2
            test_data.append(d)
            test_class.append(1)

        for x, y in group:
            d = [-2 + x, -2 + y]  # group4 -> class2
            test_data.append(d)
            test_class.append(1)
        return test_data, test_class

    def gen_test_data2(self):
        test_data = []
        test_class = []
        group = np.random.uniform(low=0, high=1, size=(50, 2))

        for x, y in group:
            d = [1 + x, 1 + y]  # group1 -> class2
            test_data.append(d)
            test_class.append(1)

        for x, y in group:
            d = [1 + x, -2 + y]  # group2 -> class1
            test_data.append(d)
            test_class.append(0)

        for x, y in group:
            d = [-2 + x, 1 + y]  # group3 -> class1
            test_data.append(d)
            test_class.append(0)

        for x, y in group:
            d = [-2 + x, -2 + y]  # group4 -> class2
            test_data.append(d)
            test_class.append(1)

        return test_data, test_class


    def mainloop(self):
        self.window.mainloop()

    def gen_data(self, n):

        X = []

        group1 = np.random.uniform(low=0, high=1, size=(n, 2))
        for x, y in group1:
            X.append([0, 1 + x, 1 + y])

        for x, y in group1:
            X.append([1, 1 + x, -2 + y])

        for x, y in group1:
            X.append([2, -2 + x, 1 + y])

        for x, y in group1:
            X.append([3, -2 + x, -2 + y])

        return X

    def question1(self):
        k = self.v.get() + 2
        data, clusters = gen_data(50)
        kmean(k, data)


    def  question2(self):
        if self.count%5 == 0:
            self.text.delete(1.0, END)
        self.count +=1
        # print("Test case ", self.count)
        training_data, training_class = self.read_training_data_type1(self.data)
        test_data, test_class = self.gen_test_data()
        # print(len(test_data))
        error1 = perceptron(training_data, training_class, test_data, test_class)
        error2 = logistic_algorithm(training_data, training_class, test_data, test_class)
        self.text.insert(END, "Test case "+str(self.count)+" - Requirement 1:\n " + "\n")
        self.text.insert(END, "Perceptron's error rate"+str(error1)+"\n")
        self.text.insert(END, "Logistic function's error rate " + str(error2) + "\n")
        self.text.insert(END,"------------\n")
        print("---------")

    def question3(self):
        if self.count%5 == 0:
            self.text.delete(1.0, END)
        self.count += 1
        self.text.insert(END, "Test case "+str(self.count)+" - Requirement 2:\n" + "\n")
        training_data, training_class = self.read_training_data_type2(self.data)
        test_data, test_class = self.gen_test_data2()
        error1 = perceptron(training_data, training_class, test_data, test_class)
        error2 = single_layer_neural_network(training_data, training_class, test_data, test_class)
        self.text.insert(END, "Perceptron's error rate " + str(error1) + "\n")
        self.text.insert(END, "Neural network's error rate " + str(error2) + "\n")
        self.text.insert(END, "------------\n")
        print("---------")

if __name__ == '__main__':
    project = Project()
    project.mainloop()

