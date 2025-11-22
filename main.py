from gelu import cifar10_data

if __name__ == "__main__":
    print("GELU - recreation")
    cifar10_data(200, 0.001, 'cpu', 'GELU')