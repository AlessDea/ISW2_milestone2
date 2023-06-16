package com.mycompany.app;

import utils.ProjectsUtils;

import java.io.*;

public class Main {

    private static int releases;
    private static String projName;

    public static void arffInit(FileWriter wr, String name) throws IOException {
        wr.write("@RELATION " + name + "\n\n");
        wr.write("@ATTRIBUTE LOCs numeric\n");
        wr.write("@ATTRIBUTE Churn numeric\n");
        wr.write("@ATTRIBUTE Age numeric\n");
        wr.write("@ATTRIBUTE WeightedAge numeric\n");
        wr.write("@ATTRIBUTE NumOfAuthors numeric\n");
        wr.write("@ATTRIBUTE Revisions numeric\n");
        wr.write("@ATTRIBUTE LOCsTouched numeric\n");
        wr.write("@ATTRIBUTE LOCsAdded numeric\n");
        wr.write("@ATTRIBUTE AvgSetSize numeric\n");
        wr.write("@ATTRIBUTE NumOfFixes numeric\n");
        wr.write("@ATTRIBUTE Buggy {false,true}\n\n");
        wr.write("@DATA\n");
    }
    public static void writeArffLine(FileWriter fileWriter, String[] val) throws IOException {
        for(int i = 3; i < val.length; i++){
            fileWriter.append(val[i]);
            if(i != val.length-1){
                fileWriter.append(",");
            }
        }
        fileWriter.append("\n");
    }



    public static void walkForward(String outputDirectoryPath, String inputFilePath){

        try {
            String outputFilePathTrain;
            String outputFilePathTest;
            String line;

            String dirPath = "WalkForward-" + projName + "/";
            new File("Output").mkdir();
            new File(outputDirectoryPath + dirPath).mkdir();

            for(int index = 2; index < releases; index++){
                outputFilePathTrain = outputDirectoryPath + dirPath + index + "/Train.arff";
                outputFilePathTest = outputDirectoryPath + dirPath + index + "/Test.arff";
                new File(outputDirectoryPath + dirPath + index).mkdir();

                FileWriter fileWriterTrain = new FileWriter(outputFilePathTrain);
                arffInit(fileWriterTrain, "Train");
                FileWriter fileWriterTest = new FileWriter(outputFilePathTest);
                arffInit(fileWriterTest, "Test");

                BufferedReader reader = new BufferedReader(new FileReader(inputFilePath));

                while ((line = reader.readLine()) != null) {
                    String[] values = line.split(",");
                    if(!values[0].equals("Version")){
                        if(Integer.parseInt(values[0]) == index){
                            // csv di testing
                            writeArffLine(fileWriterTest, values);
                        } else if (Integer.parseInt(values[0]) < index) {
                            // csv di training
                            writeArffLine(fileWriterTrain, values);
                        } else {
                            break;
                        }
                    }
                }
                reader.close();
                fileWriterTrain.close();
                fileWriterTest.close();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) throws IOException {
        String outputDirectoryPath = "Output/";

        ProjectsUtils.getInstance();

        projName = ProjectsUtils.getProjectShortNames().get(0);
        releases = Integer.parseInt(ProjectsUtils.getProjectsReleasesNumber().get(0));
        walkForward(outputDirectoryPath, ProjectsUtils.getFilesPath().get(0));

        projName = ProjectsUtils.getProjectShortNames().get(1);
        releases = Integer.parseInt(ProjectsUtils.getProjectsReleasesNumber().get(1));;
        walkForward(outputDirectoryPath, ProjectsUtils.getFilesPath().get(1));
    }
}
