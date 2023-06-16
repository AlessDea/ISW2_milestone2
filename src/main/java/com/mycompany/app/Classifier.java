package com.mycompany.app;

import weka.RetrieveInfoFromWeka;
import utils.EvaluationFile;
import utils.AllEvaluationLists;
import utils.ProjectsUtils;

public class Classifier {

    static int RELEASES;

    public static void main(String[] args) throws Exception {
        ProjectsUtils.getInstance();

        String path = "Output/WalkForward-BK/";
        String projName = ProjectsUtils.getProjectNames().get(0);
        RELEASES = Integer.parseInt(ProjectsUtils.getProjectsReleasesNumber().get(0)) -1;

        RetrieveInfoFromWeka retWekaInfo = new RetrieveInfoFromWeka(path, RELEASES);
        AllEvaluationLists allLists = retWekaInfo.retrieveClassifiersEvaluation(projName);

        EvaluationFile evaluationFileAvg = new EvaluationFile(path + projName, allLists.getAvgEvaluationsList(), "avg");
        evaluationFileAvg.csvWrite(projName);
        EvaluationFile evaluationFileDetails = new EvaluationFile(path + projName, allLists.getMergeEvaluationsList(), "details");
        evaluationFileDetails.csvWrite(projName);


        path = "Output/WalkForward-SY/";
        projName = ProjectsUtils.getProjectNames().get(1);
        RELEASES = Integer.parseInt(ProjectsUtils.getProjectsReleasesNumber().get(1)) -1;

        retWekaInfo = new RetrieveInfoFromWeka(path, RELEASES);
        allLists = retWekaInfo.retrieveClassifiersEvaluation(projName);

        evaluationFileAvg = new EvaluationFile(path + projName, allLists.getAvgEvaluationsList(), "avg");
        evaluationFileAvg.csvWrite(projName);
        evaluationFileDetails = new EvaluationFile(path + projName, allLists.getMergeEvaluationsList(), "details");
        evaluationFileDetails.csvWrite(projName);
    }
}
