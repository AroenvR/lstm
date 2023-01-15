const fs = require("fs");

/**
 * Write data to file.
 * @param fileName - name of file to write to.
 * @param data - data to write to file.
 */
export const writeToFile = async (fileName: string, data: any) => {
    fs.writeFile(fileName, JSON.stringify(data), (err: any) => {
        if (err) {
            console.log("Failed writing training data to file. Error: ", err);
            return;
        }
        console.log("Successfully wrote data to file!");
    });
}