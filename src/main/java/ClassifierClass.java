import control.RetrieveWekaInfo;
import files.EvaluationFile;
import model.AllEvaluationLists;

public class ClassifierClass {

    static int NUMRELEASES;

    public static void main(String[] args) throws Exception {
        String path = "Output/WalkForward-BK/";
        String projName = "bookkeeper";
        NUMRELEASES = 7;

       /* String path = "Output/WalkForward-SY/";
        String projName = "syncope";
        NUMRELEASES = 26;*/

        RetrieveWekaInfo retWekaInfo = new RetrieveWekaInfo(path, NUMRELEASES);
        AllEvaluationLists allLists = retWekaInfo.retrieveClassifiersEvaluation(projName);

        EvaluationFile evaluationFileAvg = new EvaluationFile(path + projName, allLists.getAvgEvaluationsList(), "avg");
        evaluationFileAvg.csvWrite(projName);
        EvaluationFile evaluationFileDetails = new EvaluationFile(path + projName, allLists.getMergeEvaluationsList(), "details");
        evaluationFileDetails.csvWrite(projName);
    }
}
