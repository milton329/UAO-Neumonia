### 📝 PR Title Convention
Use a [conventional prefix](https://www.conventionalcommits.org/) in your PR title to help classify the type of work:

| Prefix | When to use | Example |
|--------|-------------|---------|
| `feat:` | New feature work | `feat: add lead scoring model v2` |
| `fix:` | Bug fixes | `fix: correct null handling in feature pipeline` |
| `chore:` | Maintenance (doc updates, refactors, package updates) | `chore: upgrade metaflow to 2.11` |

### 📝 PR Type
- [ ] **Production Code** 🏭 - Requires ml-leads approval, linked to a Jira ticket, and an approved Change Request (CR)
- [ ] **Other** 🔍 - Peer approval is required for all non-production code, including, but not limited to: EDA, Reporting Dashboards, Infrastructure, etc.

---

### 📚 Description
<!---
Describe what is achieved as a result of this PR.
Please fill out all sections of this template — it helps reviewers,
automation, and future contributors understand your change.
-->

**Resolves:** <!-- Link the JIRA issue here -->

**CR:** <!-- Link or number of the Change Request (e.g., CR-12345). Required for Production Code. This enables automated CR linkage. -->

**Documentation Link (optional):** <!-- Link to any updated documentation here -->


#### Other changes / misc.
<!---
Ideally, there should be nothing here, but if you needed to
make changes to other pieces of code outside the scope of the
parent ticket, note them here
-->

#### Affects
<!---
Call out any global processes or other projects or people affected by this change. Use @mention.
-->

#### How it was tested
<!---
How did you test that this code works? Attach code or screenshots to help, if necessary
-->

---

### ✅ Pre-PR Checklist
This is your final check before asking for a review. Refer to this [Confluence doc](https://moveinc.atlassian.net/wiki/x/t4BTTBs) for detailed standards.

- [ ] [Typing](https://moveinc.atlassian.net/wiki/spaces/ML/pages/117244657847/Data+Science+Pre-PR+Review+Checks#Typing-Checks)
- [ ] [String checks](https://moveinc.atlassian.net/wiki/spaces/ML/pages/117244657847/Data+Science+Pre-PR+Review+Checks#String-Checks)
- [ ] [Code Structure](https://moveinc.atlassian.net/wiki/spaces/ML/pages/117244657847/Data+Science+Pre-PR+Review+Checks#Basic)
- [ ] [SQL Checks](https://moveinc.atlassian.net/wiki/spaces/ML/pages/117244657847/Data+Science+Pre-PR+Review+Checks#SQL-Checks)
- [ ] [Metaflow Checks](https://moveinc.atlassian.net/wiki/spaces/ML/pages/117244657847/Data+Science+Pre-PR+Review+Checks#System%2FMetaflow-Checks)

---

### 🚀 Deployment & Post-Merge
This section outlines the steps required to get your code into production and maintain it afterward.

> **Note**: These steps are mandatory for Production Code 🏭 but not required for Exploratory Data Analysis (EDA) 🔍 PRs. Before completing the next section, please review the [Change Management Training – Jira Workflow](https://moveinc.atlassian.net/wiki/x/xYLTgRs) to complete the required URM-related checks.

#### 📦 Deployment Checklist
- [ ] Unified Release Management (URM) Change Request (CR) is approved.
- [ ] Change Advisory Board (CAB) review is approved if required (check one):
  - [ ] Link to CAB approval: {add link}
  - [ ] CAB approval not required
- [ ] Update `ml/infra/omek_production_flows.yaml`. See [Pushing a flow to Production](https://moveinc.atlassian.net/wiki/x/EgBcaRs) for information
- [ ] Ensure dry deployment succeedes in circleci
- [ ] Ensure the correct Slack notification webhook is set. See `omek_production_flows.yaml` to know about the available webhooks.
- [ ] Monitor the progress of the deployment workflow in CircleCI. See [Navigating to CircleCI from GitHub](https://moveinc.atlassian.net/wiki/spaces/ML/pages/117731754002/Pushing+a+flow+to+Production#Navigating-to-CircleCI-from-GitHub) for more information


#### 🧹 Cleanup & Maintenance (Post-Merge)
<!---
What do you need to do after this is merged? Make sure to check them off when you do them!
-->
* [ ] **Monitor and validate data** in production to ensure the change is working as expected.
