import React from "react";

import { Category } from "@/components/Sidebar/Category";
import * as mocks from "@/mocks";
import { testUtils } from "@/utils";

describe("Category", () => {
  test("renders correctly", () => {
    const props = { category: mocks.sidebarCategory };
    const tree = testUtils
      .createSnapshotsRenderer(<Category {...props} />)
      .toJSON();
    expect(tree).toMatchSnapshot();
  });
});
