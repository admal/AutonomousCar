from model import Model

def main():
    model = Model("C:\\Users\\ASUS\\Documents\\PW\\SieciNeuronowe\\Projekt2\\Data\\driving_log.csv")
    model.train()
    model.evaluate()
    print("START")


if __name__ == "__main__":
    main()