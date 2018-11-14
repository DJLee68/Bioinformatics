from tkinter import *
import tkinter.messagebox
from tkinter.ttk import *
from tkinter.filedialog import askopenfilename
from model import *
from utilfunc import *
import toexcel
import wrapper

class MyFrame(Frame):

    def __init__(self, master, model):

        self.model = model
        label_width = 100
        Frame.__init__(self, master)
        self.a = 3
        self.master = master
        self.master.title("Biomarker Project")
        self.pack(fill=BOTH, expand=True)
        self.fs_size = 0
        # tr_open_frame
        tr_open_frame = Frame(self)
        tr_open_frame.pack(fill=X)
        Checkbutton()
        self.tr_open_btn = Button(tr_open_frame, text="Tr_Open_Btn", width=10, command=self.tr_file_open)
        self.tr_open_btn.pack(anchor=NW, padx=10, pady=0)

        self.tr_entry_name = Label(tr_open_frame, text="Not Directed", width=label_width)
        self.tr_entry_name.pack(anchor=NW, padx=10, expand=True)

        # ts_open_frame
        ts_open_frame = Frame(self)
        ts_open_frame.pack(fill=X)
        

        self.ts_open_btn = Button(tr_open_frame, text="Ts_Open_Btn", width=10, command=self.ts_file_open)
        self.ts_open_btn.pack(anchor=NW, padx=10, pady=10)

        self.ts_entry_name = Label(tr_open_frame, text="Not Directed", width=label_width)
        self.ts_entry_name.pack(anchor=NW, padx=10, expand=True)

        #제거할 고유 데이터
        unique_column_frame = Frame(self)
        unique_column_frame.pack(fill=X)
        
        unique_column_label = Label(unique_column_frame, text="고유 데이터 제거:", width=15)
        unique_column_label.pack(side=LEFT, anchor=N, padx=10, pady=10)
        
        self.unique_column_combobox = Combobox(unique_column_frame, width=20, textvariable=str)
        self.unique_column_combobox.pack(side=LEFT, anchor=N, padx=10, pady=10)
        
        unique_column_sel_btn = Button(unique_column_frame, text="해당 데이터 제거", width=15, command=self.delete_unique_column_idx)
        unique_column_sel_btn.pack(side=LEFT, anchor=N, padx=10, pady=10)
        # 데이터 타입, 클래스 선택
        data_type_frame = Frame(self)
        data_type_frame.pack(fill=X)

        data_type_label = Label(data_type_frame, text="데이터 타입:", width=15)
        data_type_label.pack(side=LEFT, anchor=N, padx=10, pady=10)

        self.type_indi_label = Label(data_type_frame, text="None", width=10)
        self.type_indi_label.pack(side=LEFT, anchor=N, padx=10, pady=10)

        class_type_label = Label(data_type_frame, text="클래스 타입: ", width=10)
        class_type_label.pack(side=LEFT, anchor=N, padx=10, pady=10)

        self.class_combobox = Combobox(data_type_frame, width=20, textvariable=str)
        self.class_combobox.pack(side=LEFT, anchor=N, padx=10, pady=10)
        self.class_sel_btn = Button(data_type_frame, text="클래스 선택", width=10, command=self.set_answer_idx)
        self.class_sel_btn.pack(side=LEFT, anchor=N, padx=10, pady=10)


        #전처리
        preprocess_type_frame = Frame(self)
        preprocess_type_frame.pack(fill=X)

        preprocess_type_label = Label(preprocess_type_frame, text="전처리 타입:", width=15)
        preprocess_type_label.pack(side=LEFT, anchor=N, padx=10, pady=10)

        self.preprocess_combobox = Combobox(preprocess_type_frame, width=20, textvariable=str)
        self.preprocess_combobox['values'] = self.model.preprocessor.get_alg_list()
        self.preprocess_combobox.pack(side=LEFT, anchor=N, padx=10, pady=10)

        preprocess_sel_btn = Button(preprocess_type_frame, text="전처리 선택", width=10, command=self.set_data_preprocess)
        preprocess_sel_btn.pack(side=LEFT, anchor=N, padx=10, pady=10)

        # 특징선택
        fs_type_frame = Frame(self)
        fs_type_frame.pack(fill=X)

        fs_type_label = Label(fs_type_frame, text="특징 선택 타입:", width=15)
        fs_type_label.pack(side=LEFT, anchor=N, padx=10, pady=10)

        self.fs_combobox = Combobox(fs_type_frame, width=20, textvariable=str)
        self.fs_combobox['values'] = self.model.fselector.get_alg_list()
        self.fs_combobox.pack(side=LEFT, anchor=N, padx=10, pady=10)

        fs_sel_btn = Button(fs_type_frame, text="특징 선택", width=10, command=self.set_data_fs)
        fs_sel_btn.pack(side=LEFT, anchor=N, padx=10, pady=10)

        # 분류기
        classifier_type_frame = Frame(self)
        classifier_type_frame.pack(fill=X)

        classifier_type_label = Label(classifier_type_frame, text="분류기 타입:", width=15)
        classifier_type_label.pack(side=LEFT, anchor=N, padx=10, pady=10)

        self.classifier_combobox = Combobox(classifier_type_frame, width=20, textvariable=str)
        self.classifier_combobox['values'] = self.model.classifier.get_alg_list()
        self.classifier_combobox.pack(side=LEFT, anchor=N, padx=10, pady=10)

        classifier_sel_btn = Button(classifier_type_frame, text="학습 시작", width=10, command=self.start_classify)
        classifier_sel_btn.pack(side=LEFT, anchor=N, padx=10, pady=10)



        # 결과
        result_frame = Frame(self)
        result_frame.pack(fill=X)

        result_accuracy_label = Label(result_frame, text="정답률: ", width=10)
        result_accuracy_label.pack(side=LEFT, anchor=NW, padx=10, pady=10)
        self.accuracy_label = Label(result_frame, text="0", width=10)
        self.accuracy_label.pack(side=LEFT, anchor=NW, padx=10, pady=10)

        result_precision_label = Label(result_frame, text="Precision: ", width=10)
        result_precision_label.pack(side=LEFT, anchor=NE, padx=10, pady=10)
        self.precision_label = Label(result_frame, text="0", width=10)
        self.precision_label.pack(side=LEFT, anchor=NE, padx=10, pady=10)

        result_frame2 = Frame(self)
        result_frame2.pack(fill=X)
        result_recall_label = Label(result_frame2, text="Recall: ", width=10)
        result_recall_label.pack(side=LEFT, anchor=NW, padx=10, pady=10)
        self.recall_label = Label(result_frame2, text="0", width=10)
        self.recall_label.pack(side=LEFT, anchor=NW, padx=10, pady=10)

        result_fbeta_label = Label(result_frame2, text="Fbeta Score: ", width=10)
        result_fbeta_label.pack(side=LEFT, anchor=N, padx=10, pady=10)
        self.fbeta_label = Label(result_frame2, text="0", width=10)
        self.fbeta_label.pack(side=LEFT, anchor=N, padx=10, pady=10)

        result_support_label = Label(result_frame2, text="Support: ", width=10)
        result_support_label.pack(side=LEFT, anchor=NW, padx=10, pady=10)
        self.support_label = Label(result_frame2, text="0", width=10)
        self.support_label.pack(side=LEFT, anchor=NW, padx=10, pady=10)

        result_frame3 = Frame(self)
        result_frame3.pack(fill=X)

        result_confmat_label = Label(result_frame3, text="Confusion Matrix: ", width=15)
        result_confmat_label.pack(side=LEFT, anchor=NW, padx=10, pady=10)
        self.confmat_label = Label(result_frame3, text="0", width=10)
        self.confmat_label.pack(side=LEFT, anchor=NW, padx=10, pady=10)
        self.tr_file_name = ""
        

    def tr_file_open(self):
        self.tr_file_name = askopenfilename(title="Choose your data file")
        self.tr_entry_name.config(text=self.tr_file_name)
        self.model.set_tr_file(self.tr_file_name)
        self.set_data_type(self.model.get_nparr_train())
        
        self.set_unique_column_combobox()
        
        toexcel.sheet_init(self.tr_file_name)
        
        
        
        
    def ts_file_open(self):
        self.ts_file_name = askopenfilename(title="Choose your data file")
        self.ts_entry_name.config(text=self.ts_file_name)
        self.model.set_ts_file(self.ts_file_name)

    def set_data_type(self, data):
        type_res = data_type_indicator(data)
        if type_res == 0:
            self.type_indi_label.config(text="Categorical")
        elif type_res == 1:
            self.type_indi_label.config(text="Numerical")
        elif type_res == 2:
            self.type_indi_label.config(text="String")
        elif type_res == 3:
            self.type_indi_label.config(text="Mixed")

    def set_class_combobox(self):
        self.class_combobox['values'] = self.model.get_fea_list()
    
    def set_unique_column_combobox(self):
        if self.model.unique_column_list == []:
            self.set_class_combobox()
        self.unique_column_combobox['values'] = self.model.unique_column_list
             
    def delete_unique_column_idx(self):
        self.unique_column_name = self.model.unique_column_list[self.unique_column_combobox.current()]
        
        self.model.delete_unique_column(np.where(self.model.fea_list == self.unique_column_name)[0][0])
        tkinter.messagebox.showinfo("Delete unique column", "Feature가 아닌 고유 데이터를 제거합니다.")
        self.set_class_combobox()
        
        

    def set_answer_idx(self):
        self.model.set_answer(self.class_combobox.current())
        tkinter.messagebox.showinfo("Zero Var Delete", "분산이 0인 특징을 제거합니다.")
        self.model.remove_var_zero()
        self.set_class_combobox()
        
        for prepro_num in range(3):
            self.model.pre_tr_data, self.model.pre_ts_data = self.model.preprocessor.preprocess(prepro_num, self.model.tr_data, self.model.ts_data)
            prepro_name = self.model.preprocessor.alg_list[prepro_num]
            for fselector_num in range(4):
                self.fs_size = 3
                self.model.set_fs_size(self.fs_size)
                self.model.fs_tr_data, self.model.fs_ts_data = self.model.fselector.start_fs(fselector_num, self.model.pre_tr_data, self.model.tr_ans, self.model.pre_ts_data, self.model.ts_ans, self.model.calg_idx)
                fselector_name = self.model.fselector.alg_list[fselector_num]
                if fselector_num is 3:
                    self.fs_size = 0
                    toexcel.set_value(prepro_name, fselector_name, self.fs_size, classifier_name, wrapper.k_num, wrapper.accuracy, wrapper.precision, wrapper.recall, wrapper.fbeta_score, wrapper.support, wrapper.conf_mat)
                    continue
                for classifier_num in range(6):
                    
                    accuracy, precision, recall, fbeta_score, support, conf_mat = self.model.start_classify(classifier_num)
                    accuracy = np.around(accuracy, 2)
                    precision = np.around(precision, 2)
                    recall = np.around(recall, 2)
                    fbeta_score = np.around(fbeta_score, 2)
                    support = np.around(support, 2)  
                    classifier_name = self.model.classifier.alg_list[classifier_num]
                    toexcel.set_value(prepro_name, fselector_name, self.fs_size, classifier_name, self.model.classifier.k, accuracy, precision, recall, fbeta_score, support, conf_mat)
        
        print("종료되었습니다. 다른 파일을 입력하세요.")
            

    def set_data_preprocess(self):
        if self.model.file_set:
            self.model.set_preprocess_data(self.preprocess_combobox.current())
        else:
            tkinter.messagebox.showwarning("파일 선택", "아직 파일이 완전히 선택되지 않았습니다.")

    def set_data_fs(self):
        if self.fs_combobox.current()==0 or self.fs_combobox.current()==1:
            self.fs_size = simpledialog.askinteger("Feature 개수", "몇 개를 선택하시겠습니까?", minvalue=1, maxvalue=len(self.model.fea_list))
            
        elif self.fs_combobox.current()==3 and not self.classifier_combobox.current() in self.model.fselector.get_alg_list():
            self.fs_size=0
        
        self.model.set_fs_size(self.fs_size)
        self.model.set_fs_data(self.fs_combobox.current())
    
        
    def start_classify(self):
        accuracy, precision, recall, fbeta_score, support, conf_mat = self.model.start_classify(self.classifier_combobox.current())
        accuracy = np.around(accuracy, 2)
        precision = np.around(precision, 2)
        recall = np.around(recall, 2)
        fbeta_score = np.around(fbeta_score, 2)
        support = np.around(support, 2)  
        
        
        self.accuracy_label.config(text=str(accuracy))
        self.accuracy_label.config(width=len(str(accuracy)))
        self.precision_label.config(text=str(precision))
        self.precision_label.config(width=len(str(precision)))
        self.recall_label.config(text=str(recall))
        self.recall_label.config(width=len(str(recall)))
        self.fbeta_label.config(text=str(fbeta_score))
        self.fbeta_label.config(width=len(str(fbeta_score)))
        self.support_label.config(text=str(support))
        self.support_label.config(width=len(str(support)))
        self.confmat_label.config(text=str(conf_mat))
        self.confmat_label.config(width=len(str(conf_mat)))

