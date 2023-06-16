import control.RetrieveWekaInfo;
import files.EvaluationFile;
import model.AllEvaluationLists;
import utils.ProjectsUtils;

public class ClassifierClass {

    static int NUMRELEASES;

    public static void main(String[] args) throws Exception {
        ProjectsUtils.getInstance();

        String path = "Output/WalkForward-BK/";
        String projName = ProjectsUtils.getProjectNames().get(0);
        NUMRELEASES = Integer.parseInt(ProjectsUtils.getProjectsReleasesNumber().get(0)) -1;

        RetrieveWekaInfo retWekaInfo = new RetrieveWekaInfo(path, NUMRELEASES);
        AllEvaluationLists allLists = retWekaInfo.retrieveClassifiersEvaluation(projName);

        EvaluationFile evaluationFileAvg = new EvaluationFile(path + projName, allLists.getAvgEvaluationsList(), "avg");
        evaluationFileAvg.csvWrite(projName);
        EvaluationFile evaluationFileDetails = new EvaluationFile(path + projName, allLists.getMergeEvaluationsList(), "details");
        evaluationFileDetails.csvWrite(projName);


        path = "Output/WalkForward-SY/";
        projName = ProjectsUtils.getProjectNames().get(1);;
        NUMRELEASES = Integer.parseInt(ProjectsUtils.getProjectsReleasesNumber().get(1)) -1;

        retWekaInfo = new RetrieveWekaInfo(path, NUMRELEASES);
        allLists = retWekaInfo.retrieveClassifiersEvaluation(projName);

        evaluationFileAvg = new EvaluationFile(path + projName, allLists.getAvgEvaluationsList(), "avg");
        evaluationFileAvg.csvWrite(projName);
        evaluationFileDetails = new EvaluationFile(path + projName, allLists.getMergeEvaluationsList(), "details");
        evaluationFileDetails.csvWrite(projName);
    }
}
