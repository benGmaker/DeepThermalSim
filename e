[1mdiff --git a/requirements.txt b/requirements.txt[m
[1mindex 2b39597..b776a79 100644[m
[1m--- a/requirements.txt[m
[1m+++ b/requirements.txt[m
[36m@@ -2,4 +2,5 @@[m [mhydra-core~=1.3.2[m
 numpy~=2.4.0[m
 matplotlib~=3.10.8[m
 omegaconf~=2.3.0[m
[31m-control~=0.10.2[m
\ No newline at end of file[m
[32m+[m[32mcontrol~=0.10.2[m
[32m+[m[32mcvxpy~=1.3.2[m
\ No newline at end of file[m
[1mdiff --git a/scr/experiments/validate_custom_mpc.py b/scr/experiments/validate_custom_mpc.py[m
[1mindex 8adfab5..ca41e07 100644[m
[1m--- a/scr/experiments/validate_custom_mpc.py[m
[1m+++ b/scr/experiments/validate_custom_mpc.py[m
[36m@@ -117,6 +117,6 @@[m [mclass ValidateCustomMPC:[m
             save_path = Path(self.run_dir, "results.png")[m
             plt.savefig(save_path)[m
             self.log.info(f"Figure saved at {save_path}")[m
[31m-        if self.cfg.experiment.show_figures:[m
[32m+[m[32m        if self.cfg.experiment.plot_figures:[m
             plt.show()[m
         plt.close()[m
\ No newline at end of file[m
