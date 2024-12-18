%-----------------------------------------------------------------------
% Job saved on 24-Jul-2024 08:53:55 by cfg_util (rev $Rev: 7345 $)
% spm SPM - SPM12 (7771)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------
matlabbatch{1}.spm.util.imcalc.input = {
                                        '/Users/gaser/Library/CloudStorage/Dropbox/GitHub/T1prep/div_Schneider2019_sub-02_T1w_defaced.nii,1'
                                        '/Users/gaser/Library/CloudStorage/Dropbox/GitHub/T1prep/mri/Schneider2019_sub-02_T1w_defaced_res-high_label.nii,1'
                                        };
matlabbatch{1}.spm.util.imcalc.output = 'div_label.nii';
matlabbatch{1}.spm.util.imcalc.outdir = {''};
matlabbatch{1}.spm.util.imcalc.expression = 'i1./(i2)';
matlabbatch{1}.spm.util.imcalc.var = struct('name', {}, 'value', {});
matlabbatch{1}.spm.util.imcalc.options.dmtx = 0;
matlabbatch{1}.spm.util.imcalc.options.mask = 0;
matlabbatch{1}.spm.util.imcalc.options.interp = 1;
matlabbatch{1}.spm.util.imcalc.options.dtype = 4;
matlabbatch{2}.spm.util.imcalc.input = {
                                        '/Users/gaser/Library/CloudStorage/Dropbox/GitHub/T1prep/Schneider2019_sub-02_T1w_defaced.nii,1'
                                        '/Users/gaser/Library/CloudStorage/Dropbox/GitHub/T1prep/div_label.nii,1'
                                        '/Users/gaser/Library/CloudStorage/Dropbox/GitHub/T1prep/mri/Schneider2019_sub-02_T1w_defaced_res-high_label.nii,1'
                                        };
matlabbatch{2}.spm.util.imcalc.output = 'output';
matlabbatch{2}.spm.util.imcalc.outdir = {''};
matlabbatch{2}.spm.util.imcalc.expression = 'i1.*(i3>0)./(i2+0.001)';
matlabbatch{2}.spm.util.imcalc.var = struct('name', {}, 'value', {});
matlabbatch{2}.spm.util.imcalc.options.dmtx = 0;
matlabbatch{2}.spm.util.imcalc.options.mask = 0;
matlabbatch{2}.spm.util.imcalc.options.interp = 1;
matlabbatch{2}.spm.util.imcalc.options.dtype = 4;
