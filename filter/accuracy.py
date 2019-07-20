import sys
import os
from PyQt5.QtWidgets import *
import numpy as np

class MainForm(QWidget):
    def __init__(self, name = 'MainForm'):
        super(MainForm,self).__init__()
        self.resultData=np.zeros(shape=[20,2])
        self.predictData=np.zeros(shape=[1000,2])
        self.start=0
        self.setWindowTitle(name)
        self.cwd = os.getcwd() 
        self.resize(300,200)   

        self.result = QLabel()
        self.result.setObjectName("Empty")
        self.result.setText("")
        # btn 1
        self.btn_chooseFile = QPushButton(self)  
        self.btn_chooseFile.setObjectName("btn_chooseFile")  
        self.btn_chooseFile.setText("select result")

        # btn 2
        self.btn_chooseFile2 = QPushButton(self)  
        self.btn_chooseFile2.setObjectName("btn_chooseFile2")  
        self.btn_chooseFile2.setText("select predict")
        
        # btn 3
        self.btn_calculate = QPushButton(self)  
        self.btn_calculate.setObjectName("btn_calculate")  
        self.btn_calculate.setText("calculate")
        
        layout = QVBoxLayout()

        layout.addWidget(self.btn_chooseFile)
        layout.addWidget(self.btn_chooseFile2)
        layout.addWidget(self.btn_calculate)
        layout.addWidget(self.result)
        self.setLayout(layout)

        self.btn_chooseFile.clicked.connect(self.slot_btn_chooseFile)
        self.btn_chooseFile2.clicked.connect(self.slot_btn_chooseFile2)
        self.btn_calculate.clicked.connect(self.slot_btn_calculate)

    def slot_btn_chooseFile(self):
        fileName_choose, filetype = QFileDialog.getOpenFileName(self,  
                                    "select file",  
                                    self.cwd,
                                    "All Files (*);;Text Files (*.txt)")

        if fileName_choose == "":
            print("\nselect cancel")
            return

        print("\nfile:")
        print(fileName_choose)
        self.resultData = np.genfromtxt(fileName_choose, delimiter=',')
        print("file type:",filetype)

    def slot_btn_chooseFile2(self):
        fileName_choose, filetype = QFileDialog.getOpenFileName(self,  
                                    "select file2",  
                                    self.cwd,
                                    "All Files (*);;Text Files (*.txt)")

        if fileName_choose == "":
            print("\nselect cancel")
            return

        print("\nfile:")
        print(fileName_choose)
        self.predictData = np.genfromtxt(fileName_choose, delimiter=',')
        print("file type:",filetype)

    def slot_btn_calculate(self):
        predictResult=0
        accuracy=0
        for index in range(len(self.predictData)):
            if self.resultData[0][1]<self.predictData[index][1]:
                self.start=index
                break
        print(self.start)
        for index in range(17):
            for index2 in range(8):
                predictResult+=self.predictData[self.start+index*38+index2][0]
                if predictResult>4:
                    predictResult=1
                else:
                    predictResult=0
            print(predictResult)
            if self.resultData[index][0]==predictResult:
                accuracy+=1
            print(accuracy)
            predictResult=0
            print('--------------')
        print(accuracy)
        self.result.setText(str(accuracy/17*100)+"%")
        

if __name__=="__main__":
    app = QApplication(sys.argv)
    mainForm = MainForm('TestQFileDialog')
    mainForm.show()
    sys.exit(app.exec_())
