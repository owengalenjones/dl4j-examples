(defproject ml-test "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [org.deeplearning4j/deeplearning4j-core "0.9.1"]
                 [me.raynes/fs "1.4.6"]
                 [org.nd4j/nd4j-native "0.9.1"]
                 [org.clojars.ds923y/nd4clj "0.1.1-SNAPSHOT" :exclusions [org.nd4j/nd4j-native]]]
  :main ^:skip-aot ml-test.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
