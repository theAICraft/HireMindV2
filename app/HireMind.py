import tempfile
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain import hub
from langchain_core.output_parsers import JsonOutputParser
# from langchin_core.runnables import Runnable

class HireMind:
    def __init__(self):
        self.read_prompt = hub.pull("aicraft/resume_details_extractor")
        self.analyse_prompt = hub.pull("aicraft/hiremind_json_score")
        self.jsonparser=JsonOutputParser()
    
    def load_model(self, model_name: str, api_key: str, openai=True):
        self.model_name = model_name
        if openai:
            self.__model = ChatOpenAI(temperature=0, model=self.model_name, 
                                      max_completion_tokens=500, api_key=api_key)
        else:
            self.__model = ChatGroq(temperature=0.7, model=self.model_name, 
                                    model_kwargs={"max_completion_tokens": 500}, api_key=api_key, verbose=True)
    
    def test_model(self):
        if self.__model:
            try:
                test_output = self.__model.invoke("", max_completion_tokens=1)
            except Exception as e:
                raise ValueError(f"Failed to load model {self.model_name} with error: {e}")    
    
    def read_resume(self, resume_path: str):  
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
             tmp_file.write(resume_path.read())
             tmp_path = tmp_file.name

        resume = PyPDFLoader(tmp_path)
        documents=resume.load()
        resume_text = "\n".join([doc.page_content for doc in documents])
        self.__chain = self.read_prompt | self.__model | self.jsonparser
        out=self.__chain.invoke({"resume_text": resume_text})
        return out
    
    def analyse_resume(self, resume_content, jd_content):
        self.__chain = self.analyse_prompt | self.__model | self.jsonparser
        out=self.__chain.invoke({"JD_TEXT_HERE":jd_content,
              "RESUME_TEXT_HERE":resume_content})
        return out