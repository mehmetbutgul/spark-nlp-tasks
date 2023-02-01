package tasks

import org.apache.spark.sql.SparkSession

object SparkUtil {
  def getSession(): SparkSession = {
    val spark= SparkSession.builder().master("local[*]").appName("tasks").getOrCreate()
    spark.sparkContext.setLogLevel("Error")
    spark
  }

}
